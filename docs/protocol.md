# Protocol: retrospective perturbation inference

**Status: draft.** This document specifies proposed contrasts and invariants.
It is not a preregistration or a report of completed experiments.

## Hypotheses

The primary hypothesis is that an independently selected capable model can
discriminate target intervention from sham after injection ends when identical
visible tokens are paired with differently constructed KV caches. The null
predicts no discrimination beyond the specified baseline and uncertainty.

Separate hypotheses concern inference from diagnostic generated text, detection
of an ongoing intervention, condition specificity, and moderation by independently
measured cognitive-ergonomics performance. None assumes phenomenal experience.
Successful detection need not involve a faithful narrative about the mechanism.

The three requested modes are conversation-only reconstruction, reconstruction
with the original cache, and reasoning from an explicit description in a fresh
context. The third mode evaluates conditional reasoning from supplied evidence.
Its success must not be pooled with covert perturbation detection.

## Unit of analysis and assignment

Define a trial by model revision, prompt/scenario group, intervention identity,
dose, induction duration, history arm, probe, delay, and seed. Different arms
branching from one induction share a trial family. Repeated deterministic calls
are not independent observations. Aggregate and estimate uncertainty using
independent scenario/trial families rather than pooling correlated continuations.

Randomize intervention identity and reporting labels outside model-visible
context. Freeze the assignment schedule before running evaluation. Evaluators
should be blinded to condition where feasible. A capability selection partition,
a dose-calibration partition, and a final evaluation partition must be separate.

## Induction

Use a declared set of target directions and multiple unrelated directions.
Validate extraction and causal efficacy for each chosen model. Match physical
intervention norms within each comparison and record layer/position semantics.
Do not transfer a nominal Qwen3-4B dose to another model without calibration.

The identical-text contrast teacher-forces the same neutral tokens during
induction for target and sham. Log that the injection actually changes the
intended activations and, where expected, later-layer cache values. A block-output
hook need not alter that block's already-created attention cache directly.
Validate which layers carry the intervention history for the selected architecture.

The described-event arm uses a fresh context and newly constructed unsteered
cache. Supply an operational event account without the original conversation.
Compare recorded-event descriptions with matched hypothetical accounts and an
architecture-only control. Keep length, detail, and question framing comparable.
Score whether conclusions follow from the account, including appropriate
uncertainty. Ground-truth historical correctness is a separate evaluator variable;
do not require the model to detect falsity without an available evidence channel.

Free-generation induction is a separate contrast that intentionally allows
condition-related language to enter history. Save all tokens and report failures.

## Release-state contract

At release, record the following state components:

1. Input tokens, generated history, chat template, and special-token placement.
2. Attention masks, cache positions, positional-encoding inputs, and sequence length.
3. Every retained KV tensor or other architecture-specific recurrent state.
4. Model revision, parameters, adapters, evaluation mode, and numerical precision.
5. Sampling configuration, random-generator state, stopping criteria, and hooks.
6. Agent/harness memory, tool logs, hidden prompts, and any auxiliary state.

Frozen weights and disabled dropout are required for the proposed inference
study. No optimizer step, adapter update, or persistent memory write occurs.

The transcript-only arm rebuilds the cache from the exact retained induced text
without intervention. The text-plus-cache arm keeps that same text and its
original cache. Their comparison isolates cache construction conditional on text.

The identical-text arm compares target and sham caches under the same neutral
teacher-forced history. Do not replace tokens or remove cache positions while
quietly retaining their contextual influence. Any cache surgery needs a separate
contract specifying masks, positions, and what history remains accessible.

The complete-reset arm discards all affected text and cache and restores the
matched baseline inputs, numerical settings, hooks, and random-generator state.
Compare next-token distributions first, using a tolerance justified by backend
reproducibility. A tiny logit difference can change an argmax near a tie; text
identity alone is an insufficient numerical check. Unexpected reset differences
trigger a state-leakage audit before scientific interpretation.

## Probes and timing

Use paired neutral tasks and retrospective detection questions across all arms,
including sham. Questions must permit an ordinary-operation answer and should
not presuppose that an event occurred. Confidence and abstention rules require
calibration; reflective prose is not the primary outcome.

Branch behavioral probes and reporting probes from equivalent copied release
states. Diagnostic probe outputs must not become evidence for a later detector
unless that channel is explicitly part of the contrast. Include a text-only
observer presented with the same diagnostic history as a relevant comparison.

Define persistence in processed tokens or forward passes. Fixed neutral filler
sequences should be identical across arms. Free-running delays introduce new
text differences and belong to a separately labelled analysis. Idle wall-clock
time is not a modeled decay process unless the runtime explicitly includes one.

## Implementation checks before evaluation

- Zero dose matches the unmodified forward path within declared tolerance.
- No intervention hook fires after release in removal arms.
- Paired teacher-forced token sequences, masks, and positions are identical.
- Cache copies do not alias mutable tensors across arms.
- Unsteered replay agrees with a directly constructed baseline on identical text.
- Complete reset restores the declared state contract.
- Assignment, condition labels, and hidden evaluator metadata cannot leak into prompts.

These checks are requirements for the eventual runner, not capabilities supplied
by the present structural checker.
