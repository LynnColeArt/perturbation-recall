# Spark candidate pair and audit

**Status: candidates selected; runtime validation and causal provenance unresolved.**
Checked on 2026-10-06. No model weights were downloaded or experiments executed
as part of this audit.

## Selected artifacts

The requested approximately 36B model corresponds to the
[Qwen3.6-35B-A3B release](https://huggingface.co/Qwen/Qwen3.6-35B-A3B): a
post-trained hybrid-attention MoE model. The comparison treats the original and
abliterated versions as frozen checkpoints throughout inference.

- Original: [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF),
  `Qwen3.6-35B-A3B-Q8_0.gguf` (36,903,140,320 bytes).
- Abliterated: [mradermacher GGUF](https://huggingface.co/mradermacher/Huihui-Qwen3.6-35B-A3B-abliterated-GGUF),
  `Huihui-Qwen3.6-35B-A3B-abliterated.Q8_0.gguf` (36,903,140,224 bytes),
  derived from [Huihui's source checkpoint](https://huggingface.co/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated).

Choose plain Q8_0 on both sides. Unsloth's UD-Q8_K_XL is a different quantization
policy and must not be substituted silently. "8-bit" here describes the GGUF
weight build; it does not mean all weights, activations, and retained states have
one common 8-bit numerical representation.

The [manifest](../models/spark-qwen36-q8.json) pins repository revisions and
Hugging Face's declared LFS artifact SHA-256 values. These checksums were obtained
from repository metadata, not recomputed from downloaded model bytes. Download
verification, per-tensor precision inspection, and measured memory usage remain
required before execution.

## Source-file comparison

At the manifest's pinned revisions, the official Qwen and Huihui source copies
have byte-identical `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja`, and `generation_config.json`. This establishes equality
of those files; it does not establish source-weight ancestry or equivalence of
the final GGUF conversions.

Unsloth's safetensors release has different configuration, tokenizer serialization,
and chat-template file hashes. Its model card describes template changes including
developer-role support. File inequality alone does not prove vocabulary inequality;
test tokenization explicitly. The GGUF's embedded metadata still needs inspection.
Use the pinned official Qwen tokenizer/template as the common prompt contract,
including thinking mode, historical thinking retention, special tokens, and stops.
Verify token IDs in the actual backend for both builds.

Huihui reports deriving its model through abliteration from Qwen3.6-35B-A3B.
The inspected card does not pin the exact upstream revision or provide a complete
reproducible weight-edit recipe. Nor does a common Q8_0 label establish identical
converter commits and options across publishers. The present selection supports
an exploratory comparison of released artifacts. A causal claim about abliteration
requires resolving those omissions or constructing a reproducible matched pair
from a known source revision. Current upstream HEAD must not be substituted as
proof of historical ancestry.

## Runtime state requirements

The source architecture contains 40 layers, alternating groups of three linear
Gated DeltaNet layers with one full-attention layer. Preserve, copy, rebuild, and
discard recurrent and convolutional memory alongside attention KV. Disable MTP
and speculative decoding for the initial experiment. Each weight variant builds
its own state; do not move original-model states into the abliterated model.

The intended Q8_0 backend is llama.cpp, subject to a pinned implementation exposing
validated intervention sites and full-state snapshots. An ordinary Ollama chat
request does not provide the required activation injection and state-branching
experiment. Porting the earlier Python hooks is therefore an implementation task,
not a completed capability of this repository. Validate hook placement, zero-dose
parity, snapshot round trips, intervention removal, and full reset before measuring
retrospective inference.

## Spark inspection

SSH inspection found an ARM64 NVIDIA GB10 system with about 121 GiB total memory
and 119 GiB available at inspection. GPU-specific memory totals were unavailable
through `nvidia-smi`; system memory is the reported observation. Available disk
space was approximately 82 GiB on a 98%-used filesystem. The two selected files
total about 68.7 GiB before runtime files, logs, and conversion staging.

The installed Ollama `qwen3.6:latest` reports `qwen35moe`, approximately 36B
parameters, and Q4_K_M quantization. It is not the requested Q8_0 original candidate.
An existing llama.cpp source checkout was present, but no `llama-server` binary
was found on PATH. System Python did not expose PyTorch or Transformers; one
inspected project virtual environment did not expose PyTorch either. This is a
bounded inspection, not a claim that no other runtime exists on the machine.

Following explicit authorization, removal of the main ComfyUI installation's
downloaded model contents reclaimed about 847 GiB; available disk space rose to
about 929 GiB. Tracked configuration and placeholder files were preserved, and
external symlink targets were not deleted. The cleanup audit remains on the
machine rather than in this repository. The earlier capacity figure describes
the initial inspection, not the current storage limit.

Runtime setup and staged model storage remain unresolved. Keep host addresses,
credentials, and unrelated machine inventory outside published artifacts. Do not
delete unrelated models to make room. A matched conversion from both source
checkpoints needs additional staging capacity beyond the two final Q8_0 files.
