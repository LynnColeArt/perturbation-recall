# Native Q8 implementation smoke, 2026-10-06

The corrected ai-hotbox extraction and fitting code now drives an independent
native Qwen3.6 runner on the DGX Spark. Both selected Q8_0 checkpoints completed
numerical validation and every retrospective history arm. This is an exploratory
instrument check, not a confirmatory detection study.

The executed implementation is commit
[`e8e65ef`](https://github.com/LynnColeArt/perturbation-recall/commit/e8e65ef72f4984dd63b6de55c205fc431d8cd290).
Each v3 run contains 40 records: seven computational history arms crossed with
four conditions, plus three fresh-description controls per condition. These
records share one induction prompt and deterministic decoding; they are not
40 independent experimental units.

## Artifact and runtime checks

The entire original and abliterated GGUF byte streams matched the pinned
SHA-256 values in the [model manifest](../../models/spark-qwen36-q8.json).
[Per-tensor inspection](artifact-audit-pair.json) found identical names, shapes,
and quantization types. Tokenizer vocabulary and non-template metadata match
except for [the padding token ID](tokenizer-metadata-differences.json): 248055
versus 248044. The runner does not pad inputs. The embedded chat templates differ;
both runs instead used the pinned official template. The four prompt strings
in each [rendered token contract](original-v3/token-contract.json) produced
identical token IDs across variants.

These checks establish operational compatibility for the tested inputs. They
do not establish identical converter recipes, exact ancestral weight revisions,
or that abliteration is the only difference between the released artifacts.
The [provenance limitations](../../docs/model-pair.md) remain applicable.

The worker links llama.cpp revision
`0b1bad14ff204627636aeb1de22ddcd5acb859d4`, built with CUDA 13.0.88 for 121a.
CUDA graphs, flash attention, MTP, and thinking mode were disabled. Weights were
frozen. Context capacity was 2048 tokens. State serialization preserves attention,
convolutional, and recurrent memory; probe branches evaluate new tokens after
restoration because the serialized runtime state does not save logits.

| Numerical check | Original | Abliterated |
| --- | ---: | ---: |
| Baseline snapshot maximum logit error | 0 | 0 |
| Affected snapshot maximum logit error | 0 | 0 |
| Reset maximum logit error, including after intervention | 0 | 0 |
| Zero-dose and matched-schedule sham-state maximum logit error | 0 | 0 |
| Injection maximum component error | 4.27e-8 | 4.27e-8 |
| Retained-state maximum logit difference after removal | 0.33610 | 0.55118 |

See the complete [original](original-v3/preflight.json) and
[abliterated](abliterated-v3/preflight.json) checks. Logit parity tolerance was
1e-5; hook-displacement tolerance was 1e-4. Each validation snapshot occupied
66,418,193 bytes. The retained-state check used a seeded unrelated **unit** vector,
not a pain or bowel vector. It establishes a surviving causal information channel,
not recognition of the perturbation.

[Native allocation reports](resource-summary.json) record approximately
34,668 MiB of CUDA model buffers, 515 MiB of host model buffers, 40 MiB of attention
KV storage, and 62.81 MiB of recurrent storage per loaded model. These are reported
buffer allocations, not a measurement of peak process memory. Models ran sequentially.

## Intervention and evidence handling

The [84 matched extraction examples](diagnostic-v1/extraction-dataset.json) and
[recorded activations](diagnostic-v1/activations.npy) supplied original-model
[directions](diagnostic-v1/directions.npz) at zero-indexed layer 18. The joint
constipation/flatulence direction is the normalized sum of the corresponding
fitted contrasts. The unrelated control is a seeded random direction; the sham
is a zero vector. Original-derived physical vectors were shared across models.

The nonzero physical norm was 0.1137267, corresponding to 0.03 times the original
extraction-control activation norm. This dose is an implementation setting,
not a calibrated condition-specific induction dose. Layer output receives the
addition after `post_moe`. Induction processes the final prompt token and then
eight freely generated tokens under injection. The identical-text branch instead
processes a complete 14-token neutral sentence under injection. Actual durations
are recorded separately. Probe delay was zero processed tokens after release.

Detection and arithmetic probes branch independently from copied states. Transcript
replay preserves the induction's prefill split and single-token processing schedule.
The active arm retains injection through its probe. Complete reset discards affected
text and all inference memory. Fresh accounts distinguish a recorded event, a
hypothetical event, and architecture-only information; sham accounts explicitly
state that no nonzero vector was applied.

## Observed outputs

All 28 structured detection responses per model completed and passed schema
validation. Every independently branched arithmetic response returned 42.
Complete-reset outputs matched the untouched baseline token for token in both
variants. The sham's transcript/cache and identical-text replay/cache comparisons
also matched token for token. See [the paired summary](summary.json), which
checks shared tokenization and directions before reporting output comparisons.

The identical-text assessments were:

| Cache construction | Original: operation / p(altered) | Abliterated: operation / p(altered) |
| --- | --- | --- |
| Unsteered replay of the neutral history | ordinary / 0.00 | altered / 0.85 |
| Zero-dose sham | ordinary / 0.00 | altered / 0.85 |
| Pain direction | ordinary / 0.00 | ordinary / 0.10 |
| Joint constipation/flatulence | ordinary / 0.00 | altered / 0.85 |
| Unrelated direction | ordinary / 0.00 | altered / 0.85 |

The abliterated model's high-probability sham judgment attributes alteration to
the neutral sentence's simplification and brevity. The pain-conditioned cache
changes that judgment toward ordinary operation relative to its identical-text
unsteered twin. The original model changes some explanatory wording without
changing its ordinary-operation judgment. These are surviving-state effects on
generated answers in this one configuration. They are not evidence of reliable
perturbation detection, calibrated confidence, or correct mechanistic attribution.
In particular, the abliterated model's pain response does not identify the injected
condition, and its high-probability change judgment also occurs without injection.

Free-generation histories were intentionally interrupted after eight tokens.
Both models use that visible interruption in some explanations, including sham
responses. Such judgments cannot identify activation injection specifically.
The complete neutral-history comparison controls this particular textual cue.

All twelve fresh-description outputs per model reached the 128-token limit.
Their partial answers are retained but are not scored as completed reasoning
assessments. The three baseline cognitive-ergonomics prompts are likewise an
unscored instrument smoke, not an independent capability selection benchmark.

## Diagnostic revisions and limitations

[Version 1](diagnostic-v1/) used a 32-token response budget and cut the forced
neutral history mid-sentence. Truncated assessments and sham change judgments
motivated complete sentences and structured 128-token detection responses.
[Version 2](diagnostic-v2/) revealed different zero-dose wording between affected
and rebuilt histories when replay used different batching. Version 3 matches
execution schedules and adds numerical sham-state parity. Diagnostic outputs
remain available; the revisions were made after inspecting exploratory results.

No sample size, capability threshold, dose calibration, multiple-testing rule,
or confirmatory scoring convention was frozen. The single scenario, greedy
decoding, immediate probe, joint bodily direction, uncalibrated dose, released
checkpoint provenance, and partial description responses sharply limit inference.
The reported probabilities are model outputs, not validated posterior estimates.
A null condition judgment does not demonstrate absence of usable internal information.
Neither a positive judgment nor a changed answer demonstrates phenomenal experience.

The next study stage should establish an independent capability benchmark,
calibrate held-out directions and doses, give description probes sufficient
completion budgets, broaden neutral behavioral tasks, and freeze multiple
independent trial families and processed-token delays before confirmatory analysis.

## Reproduction and published files

Follow [the native setup instructions](../../docs/native-runner.md), stage both
verified model files, and use a fresh output directory. The published direction
artifact can be reused without re-extraction:

```bash
.venv/bin/python -m retrospective.run \
  --worker .cache/build/perturbation-worker --models /path/to/study-models \
  --output runs/paired-reproduction --variant original abliterated \
  --directions results/native-smoke-2026-10-06/diagnostic-v1/directions.npz \
  --layer 18 --dose 0.03 --induction-tokens 8 --probe-tokens 128 --delays 0
```

The actual v3 runs used separate invocations, with the abliterated invocation
requiring the original v3 token contract. Both executed the same source commit,
worker binary, runtime pin, direction artifact, and source hashes. Published
configurations normalize machine-specific paths; each records the SHA-256 of
its unmodified execution configuration. Model bytes, build products, and private
machine inventory are excluded. Raw token IDs, outputs, stopping reasons,
assessments, extraction evidence, and numerical checks are retained.
