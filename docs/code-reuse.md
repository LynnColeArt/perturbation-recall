# Reusing the ai-hotbox implementation

This repository starts from the corrected ai-hotbox implementation at
[`a0f63f0c`](https://github.com/LynnColeArt/ai-hotbox/tree/a0f63f0c2806c3dc91ecd418c0d54db9bbc38f72).
The imported files and their original SHA-256 hashes are recorded in
[code-provenance.json](../models/code-provenance.json). The source
[license and attribution](../licenses/ai-hotbox-LICENSE) are retained.
Derivative prompts, vectors, and protocols credit **[the Saw Test](https://clanker.church)**.

## Reused components

| Component | Existing behavior | Role in the new study |
| --- | --- | --- |
| `impossible_states/dataset.py` | Matched constipation × flatulence scenarios, pain and nuisance controls, scenario-level partitions | Exploratory extraction materials; not a validated retrospective detection instrument |
| `impossible_states/analysis.py` | Factorial vector extraction, interaction direction, nuisance-span adjustment, AUC, lexical/repetition metrics | Direction fitting and secondary diagnostics; lexical hits are not the primary detection outcome |
| `impossible_states/engine.py` | Frozen-weight extraction, final-position hooks, continuous/pulse/rebuild/opposing interventions, seeded sampling, teacher-forced replay | Dense-Qwen3 reference implementation for intervention and release behavior |
| `tests/test_impossible_states.py` | Synthetic factorial checks and tiny randomly initialized Qwen3 cache checks | Preserve the previously checked behavior while adapting the backend |

The reference engine has one local change: it rejects other architectures before
accessing their model internals. Qwen3.6's hybrid recurrent memory is not validated
by a dense-Qwen3 regression test. The original command-line runner's Qwen3-4B
defaults, the fixed-layer chamber script, and unrelated repository contents are
not imported as the new study's execution interface.

## Verify the inherited behavior

In a Python environment containing PyTorch and the versions in
`requirements-reference.txt`, run:

```bash
python -m unittest discover -s tests -v
python scripts/check_protocol.py
```

The numerical cache tests construct a tiny random model on CPU; they do not
download weights or generate scientific evidence about the selected checkpoints.
The dependency file records the reference environment, not an ARM64 Spark setup
or support for Qwen3.6. Do not install a desktop CUDA wheel on the Spark from this
reference specification.

## Adaptation work

Retain the vector-fitting logic, intervention schedule, transcript-replay logic,
and causal checks. Adapt model access and the hook implementation to the selected
Q8_0 backend, then validate the exact intervention site and precision. The prior
engine does not expose release-state snapshots or separate retrospective probes;
those interfaces must be added for branching and neutral post-release delays.

Preserve and reset the full hybrid memory object, with explicit position/mask
semantics and independent state copies. Add reporting and behavioral probe branches,
the fresh-description arm, matched variant orchestration, and structured detection
and confidence scoring under the new protocol. Use a shared original-model
direction for the initial weight comparison and keep native re-extraction separate.

The [native runner](native-runner.md) adapts those mechanisms through the pinned
llama.cpp control-vector and full-memory serialization APIs. It preserves the
inherited fitting code and adds release-state branching and retrospective probes.
Run its zero-dose, release, snapshot round-trip, teacher-forced replay, and complete-reset
checks on each selected artifact before an exploratory run.
