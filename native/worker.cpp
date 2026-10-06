#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

struct capture_data {
    std::set<int> layers;
    std::map<std::string, std::vector<float>> values;
    std::string error;
};

static bool capture(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & data = *static_cast<capture_data *>(opaque);
    const std::string name = ggml_get_name(tensor);
    bool wanted = false;
    for (int layer : data.layers) {
        if (name == "l_out-" + std::to_string(layer) || name == "post_moe-" + std::to_string(layer)) {
            wanted = true;
        }
    }
    if (ask || !wanted) return wanted;
    if (tensor->type != GGML_TYPE_F32 || tensor->ne[2] != 1 || tensor->ne[3] != 1 || tensor->nb[0] != sizeof(float)) {
        data.error = "Unsupported activation tensor layout: " + name;
        return false;
    }
    auto & values = data.values[name];
    values.resize(tensor->ne[0]);
    const size_t offset = (tensor->ne[1] - 1) * tensor->nb[1];
    ggml_backend_tensor_get(tensor, values.data(), offset, values.size() * sizeof(float));
    return true;
}

struct snapshot {
    std::vector<uint8_t> bytes;
    std::vector<llama_token> tokens;
    std::vector<float> direction;
    int layer = -1;
};

class worker {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    capture_data captured;
    std::vector<llama_token> tokens;
    std::vector<float> direction;
    int steering_layer = -1;
    std::map<std::string, snapshot> snapshots;

    void set_steering(const std::vector<float> & vector, int layer) {
        const int width = llama_model_n_embd(model);
        const int layers = llama_model_n_layer(model);
        if (vector.empty()) {
            if (llama_set_adapter_cvec(ctx, nullptr, 0, width, -1, -1)) throw std::runtime_error("Cannot remove intervention");
        } else {
            if (layer < 1 || layer >= layers - 1 || vector.size() != size_t(width)) throw std::runtime_error("Intervention requires a nonfinal layer >= 1 and matching vector width");
            for (float v : vector) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite intervention");
            std::vector<float> controls(size_t(width) * (layers - 1), 0.0f);
            std::copy(vector.begin(), vector.end(), controls.begin() + size_t(width) * (layer - 1));
            if (llama_set_adapter_cvec(ctx, controls.data(), controls.size(), width, layer, layer)) throw std::runtime_error("Cannot install intervention");
        }
        direction = vector;
        steering_layer = vector.empty() ? -1 : layer;
    }

    std::string piece(llama_token token) const {
        std::vector<char> buffer(256);
        int n = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
        if (n < 0) {
            buffer.resize(-n);
            n = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
        }
        if (n < 0) throw std::runtime_error("Detokenization failed");
        return std::string(buffer.data(), n);
    }

public:
    worker(const char * path, int context, int gpu_layers) {
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu_layers;
        mp.load_mtp = false;
        model = llama_model_load_from_file(path, mp);
        if (!model) throw std::runtime_error("Model loading failed");
        vocab = llama_model_get_vocab(model);
        auto cp = llama_context_default_params();
        cp.n_ctx = context;
        cp.n_batch = 256;
        cp.n_ubatch = 256;
        cp.n_seq_max = 1;
        cp.n_rs_seq = 0;
        cp.n_threads = 8;
        cp.n_threads_batch = 8;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.cb_eval = capture;
        cp.cb_eval_user_data = &captured;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("Context creation failed");
    }

    ~worker() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
    }

    json dispatch(const json & request) {
        const auto op = request.at("op").get<std::string>();
        if (op == "info") {
            char architecture[128];
            llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture));
            return {{"architecture", architecture}, {"width", llama_model_n_embd(model)},
                    {"layers", llama_model_n_layer(model)}, {"vocab_size", llama_vocab_n_tokens(vocab)},
                    {"context", llama_n_ctx(ctx)}, {"mtp_loaded", false}};
        }
        if (op == "tokenize") {
            const std::string text = request.at("text");
            std::vector<llama_token> result(text.size() + 16);
            const bool special = request.value("add_special", false);
            int n = llama_tokenize(vocab, text.data(), text.size(), result.data(), result.size(), special, true);
            if (n < 0) {
                result.resize(-n);
                n = llama_tokenize(vocab, text.data(), text.size(), result.data(), result.size(), special, true);
            }
            if (n < 0) throw std::runtime_error("Tokenization failed");
            result.resize(n);
            return {{"tokens", result}};
        }
        if (op == "detokenize") {
            std::string text;
            for (auto token : request.at("tokens").get<std::vector<llama_token>>()) text += piece(token);
            return {{"text", text}};
        }
        if (op == "steer") {
            set_steering(request.value("direction", std::vector<float>{}), request.value("layer", -1));
            return {{"steering_layer", steering_layer}};
        }
        if (op == "clear") {
            set_steering({}, -1);
            llama_memory_clear(llama_get_memory(ctx), true);
            tokens.clear();
            captured.values.clear();
            return {{"position", 0}};
        }
        if (op == "snapshot") {
            llama_synchronize(ctx);
            snapshot saved;
            saved.bytes.resize(llama_state_get_size(ctx));
            const auto written = llama_state_get_data(ctx, saved.bytes.data(), saved.bytes.size());
            if (!written) throw std::runtime_error("State serialization failed");
            saved.bytes.resize(written);
            saved.tokens = tokens;
            saved.direction = direction;
            saved.layer = steering_layer;
            const auto name = request.at("name").get<std::string>();
            snapshots[name] = std::move(saved);
            return {{"bytes", written}, {"position", tokens.size()}, {"steering_layer", steering_layer}};
        }
        if (op == "restore") {
            const auto & saved = snapshots.at(request.at("name").get<std::string>());
            set_steering({}, -1);
            llama_memory_clear(llama_get_memory(ctx), true);
            if (llama_state_set_data(ctx, saved.bytes.data(), saved.bytes.size()) != saved.bytes.size()) throw std::runtime_error("Incomplete state restoration");
            tokens = saved.tokens;
            set_steering(saved.direction, saved.layer);
            captured.values.clear();
            return {{"position", tokens.size()}, {"steering_layer", steering_layer}};
        }
        if (op == "drop") {
            snapshots.erase(request.at("name").get<std::string>());
            return {{"ok", true}};
        }
        if (op == "eval") {
            const auto input = request.at("tokens").get<std::vector<llama_token>>();
            if (input.empty()) throw std::runtime_error("Empty evaluation input");
            if (tokens.size() + input.size() > llama_n_ctx(ctx)) throw std::runtime_error("Context limit exceeded");
            if (!direction.empty() && input.size() != 1) throw std::runtime_error("Steered evaluation must contain exactly one token");
            captured.layers.clear();
            for (int layer : request.value("capture_layers", std::vector<int>{})) {
                if (layer < 0 || layer >= llama_model_n_layer(model)) throw std::runtime_error("Invalid capture layer");
                captured.layers.insert(layer);
            }
            captured.values.clear();
            captured.error.clear();
            for (size_t start = 0; start < input.size(); start += 256) {
                const size_t n = std::min(size_t(256), input.size() - start);
                auto batch = llama_batch_init(n, 0, 1);
                batch.n_tokens = n;
                for (size_t i = 0; i < n; ++i) {
                    batch.token[i] = input[start + i];
                    batch.pos[i] = tokens.size() + i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = i == n - 1;
                }
                const int result = llama_decode(ctx, batch);
                llama_batch_free(batch);
                if (result || !captured.error.empty()) throw std::runtime_error("Decode failed: " + std::to_string(result) + " " + captured.error);
                tokens.insert(tokens.end(), input.begin() + start, input.begin() + start + n);
            }
            for (int layer : captured.layers) if (!captured.values.count("l_out-" + std::to_string(layer))) throw std::runtime_error("Requested layer output was not observed");
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits) throw std::runtime_error("Missing logits");
            const int size = llama_vocab_n_tokens(vocab);
            const int next = std::max_element(logits, logits + size) - logits;
            json response = {{"position", tokens.size()}, {"next_token", next},
                {"next_is_eog", llama_vocab_is_eog(vocab, next)}, {"activations", captured.values},
                {"steering_layer", steering_layer}};
            if (request.value("logits", false)) response["logits"] = std::vector<float>(logits, logits + size);
            return response;
        }
        throw std::runtime_error("Unknown operation: " + op);
    }
};

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: perturbation-worker MODEL CONTEXT GPU_LAYERS\n";
        return 2;
    }
    try {
        worker engine(argv[1], std::stoi(argv[2]), std::stoi(argv[3]));
        std::cout << json({{"ready", true}, {"info", engine.dispatch({{"op", "info"}})}}).dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                std::cout << engine.dispatch(json::parse(line)).dump(-1, ' ', false, json::error_handler_t::replace) << std::endl;
            } catch (const std::exception & error) {
                std::cout << json({{"error", error.what()}}).dump() << std::endl;
            }
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
