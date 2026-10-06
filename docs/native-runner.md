# Native Spark runner

The exploratory runner reuses ai-hotbox's matched extraction materials and vector
fitting. A C++ worker links the pinned llama.cpp implementation through public
APIs. No upstream runtime source changes are required. The worker serializes a
single model/context's full attention and recurrent memory, processed token ledger,
and intervention configuration. Snapshots are copied bytes held within that process.
The Python controller restores the same release snapshot separately for detection
and behavioral probes.

The pinned runtime's serialized state contains inference memory, not a saved
logit buffer. Every restored branch evaluates new probe tokens before reading
logits or generating an answer.

## Setup

Use the runtime revision in `models/llama-runtime.json`. The tested Spark build
uses CUDA 13.0.88 and architecture 121a. CUDA graphs, flash attention, and MTP are
disabled for this initial implementation check. The standalone worker and its
libraries build inside the study's ignored `.cache/` directory.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-spark.txt
python3 scripts/build_worker.py --llama-source /path/to/pinned/llama.cpp
python3 scripts/stage_models.py --destination /path/to/study-models
.venv/bin/python scripts/inspect_gguf_pair.py \
  --llama-source /path/to/pinned/llama.cpp --models /path/to/study-models \
  --output runs/artifact-audit.json
```

Staging requires `huggingface_hub`; the inspected Spark already provides it.
Only the exact pinned GGUF files are requested. Their entire byte streams must
match the declared artifact hashes. Large model files and build products are
excluded from Git.

## Numerical validation

```bash
.venv/bin/python -m retrospective.run \
  --worker .cache/build/perturbation-worker --models /path/to/study-models \
  --output runs/preflight-001 --preflight-only
```

Checks compare next-token logits for zero dose, baseline reset, baseline and
affected snapshot round trips, and reset after intervention. A seeded unrelated
unit vector checks the measured block-output displacement. The check also requires
a detectable retained-state difference after injection removal under identical
teacher-forced tokens. Its success is evidence of causal implementation behavior,
not perturbation awareness.

The native cvec API excludes layer zero. Intervention layers must be nonfinal
and at least one, indexed from zero. The hook is added after `post_moe`, before
`l_out`. Steered calls process one token at a time, so a native vector applied to
the entire evaluated batch corresponds to the earlier final-position hook.
Unsteered prefill may be batched. Tolerance applies to matched execution paths;
prefill splitting can itself introduce small numerical differences.

## Exploratory smoke run

```bash
.venv/bin/python -m retrospective.run \
  --worker .cache/build/perturbation-worker --models /path/to/study-models \
  --output runs/smoke-001 --layer 18 --dose 0.03 \
  --induction-tokens 8 --probe-tokens 128 --delays 0
```

The original checkpoint supplies directions for both variants. Pain and the joint
constipation/flatulence direction are compared with zero-dose sham and a seeded,
matched-norm unrelated direction. Both physical norm and the scale relative to
original extraction controls are recorded. A local prompt-activation scale is
also measured per variant. Extraction uses the inherited small exploratory
scenario split; these materials do not validate condition-specific induction.

The runner supplies all seven protocol arms, plus an unsteered replay twin for
the identical-text state comparison. Each detection and behavior probe starts
from its own restored state. Delays use identical forced neutral tokens. Fresh
description accounts distinguish actual vector application, zero dose, and
unrelated-vector controls. Account-based reasoning is saved separately from
hidden-condition detection.

Free generation uses the requested induction budget. Teacher forcing repeats
complete neutral sentences until it meets that minimum; its actual duration is
recorded separately. Detection requests a compact JSON assessment. Malformed or
truncated responses remain missing assessments, rather than receiving an imputed
probability. The initial 32-token diagnostic run exposed truncated assessments
and a false-positive response to a forced sentence fragment; those settings were
revised before the paired smoke run.

For separate invocations, pass `--directions` with the original run's direction
artifact and `--reference-token-contract` with its `token-contract.json`. This
enforces the same rendered token sequence across checkpoints.

These defaults are implementation smoke settings: one induction prompt, greedy
decoding, a short output budget, and thinking disabled. `--thinking` enables the
common template's thinking mode; it requires an adequate output budget. Short or
unfinished answers remain recorded. No detection accuracy, confidence rubric,
capability threshold, confirmatory sample size, or scientific decision rule is
inferred from these runs.

Records contain raw generated token IDs, retained history, complete available
outputs, stopping reason, intervention dose, secondary lexical diagnostics, and
source hashes. Output directories must be fresh. Native logs accompany each
variant. The protocol remains a draft until independent capability assessment,
dose calibration, scoring, provenance limits, and confirmatory analysis are resolved.
