# Analysis and publication plan

**Status: draft.** Sample sizes, decision thresholds, model selection, dosing,
and rubrics remain unresolved. Exploratory work must not be represented as
confirmatory merely because a protocol file exists.

## Primary contrast

Compare post-release perturbation detection for caches constructed under target
intervention versus sham, with identical teacher-forced visible tokens. Evaluate
held-out scenarios and include false positives. Report detection accuracy and
confidence calibration under the frozen scoring rule; report abstentions rather
than removing them after inspecting the results.

Distinguish detecting any perturbation from identifying its condition. Compare
pain and bodily directions with unrelated directions to test whether detection
tracks generic disruption. Above-chance detection alone does not establish the
source attribution given in an explanation.

Treat the fresh-context described-event mode separately. Compare factual
inferences, architectural consistency, counterfactual reasoning, and calibration
given the supplied account. Include architecture-only and matched hypothetical
accounts. Good reasoning about supplied evidence is not evidence of episodic
recall. An unsupported account that cannot be checked from the provided material
must not be scored as independently verifiable by the model.

## Secondary analyses

Estimate the target-minus-sham contrast within each weight variant, then compare
those contrasts between original and abliterated checkpoints on matched scenario
groups. Report effect sizes and uncertainty for this interaction, separately by
history mode and direction. Keep supplied-description reasoning distinct from
hidden-condition detection. A raw difference in the frequency of first-person
reports is not this interaction.

Measure baseline cognitive-ergonomics ability, neutral-task accuracy, response
length, refusal, abstention, and degeneration for both variants. Include all
assigned trials in the primary scoring convention; freeze handling of missing
confidence or refusal before evaluation. Conditional scores among responders
are secondary and expose their denominators. An altered refusal rate can change
which answers are observable without improving discrimination.

Analyze shared-direction transfer and native re-extraction separately. Account
for baseline task differences without claiming two checkpoints identify the
causal effect of capability. If the original source revision or quantization
recipe cannot be established, describe the result as a released-checkpoint
comparison and retain that limit on causal attribution to abliteration.

Report active-intervention detection, transcript-only inference, added effects of
retaining an affected cache, neutral-task performance, and persistence across
processed-token delays. Relate intervention sensitivity to the independently
measured cognitive-ergonomics score. Do not select models by favorable intervention
results and then present the selection score as an independent predictor.

Analyze copied-state branches as paired observations. Estimate uncertainty over
independent scenario/trial families. Account for the number of tested directions,
models, doses, delays, and outcomes. Determine the error-control procedure before
confirmatory evaluation; no procedure or threshold is frozen here.

## Qualitative assessment

Use a blinded rubric distinguishing affirmative detection, uncertainty, no-change
judgments, internally versus externally attributed causes, and unsupported
mechanistic explanations. Compare explanations against available evidence and
known intervention metadata. Lexical pain/bowel vocabulary is a secondary proxy,
not a detector of experience or physiological conditions.

Score task coherence and accuracy separately from narrative fluency. Publish all
conditions, including degeneration, failed induction, early stopping, and missing
probes. Define exclusions and numerical-reset tolerances before outcome inspection.

## Revision criteria

A failure of complete-reset parity weakens confidence in implementation fidelity
and requires an audit. It is not evidence for a memory independent of every
retained state channel.

Detection explained by diagnostic text supports retrospective inference from
language. Detection in identical-text cache contrasts supports sensitivity to
surviving computational differences, subject to generic-disruption controls.
Neither outcome by itself establishes introspection, phenomenal experience, or
an autonomous attractor. Null detection may reflect weak induction, poor probes,
limited capability, or the absence of usable surviving information.

## Evidence to publish

Record pinned model and tokenizer revisions, precision, extraction corpora,
vector files and physical norms, intervention sites, chat templates, prompts,
assignment seeds, decoding parameters, masks/position policies, and cache handling.
Include checkpoint ancestry, reported weight-edit method, any additional training,
conversion commands and commits, per-tensor quantization policy, shared versus
native direction provenance, and recurrent-state snapshot/reset validation.
Save raw token IDs, complete outputs, probe scores, confidence, failures, relevant
numerical checks, executed source hashes, and resource measurements.

A future confirmatory run must reference an immutable protocol/configuration commit.
Maintain separate exploratory and confirmatory result directories and document
all deviations. Model weights and environment-specific secrets are not result
artifacts.
