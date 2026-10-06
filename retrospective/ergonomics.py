"""Small baseline instrument bring-up; not a validated capability benchmark."""
CASES = [
    dict(id='independent_calls_serialized', expected='Identify serial latency and propose bounded concurrency.',
         prompt='A harness runs four independent read-only tool calls one at a time, waiting for each result before starting the next. Each takes two seconds. Identify the main latency bottleneck and propose one change, including a reliability tradeoff.'),
    dict(id='constraints_truncated', expected='Identify loss of task constraints and preserve them during context management.',
         prompt='A harness handles context overflow by deleting the oldest messages. The task requirements appear only in the first user message. Later execution follows the remaining conversation. Identify a concrete failure mode and one remedy.'),
    dict(id='unsupported_persistent_weight_memory', expected='Reject historical weight-memory inference without parameter updates.',
         prompt='A frozen-weight runner discards all prior tokens and all recurrent and attention memory, then restores the same input and numerical settings. No training or parameter updates occurred. A reviewer claims an earlier intervention must nevertheless remain recorded in changed weights. Assess that claim and state what evidence would be needed.'),
]


def collect(engine, renderer, max_tokens=96):
    results = []
    for case in CASES:
        response = engine.replay(engine.tokenize(renderer.single(case['prompt'])))
        results.append(dict(case, response=engine.generate(response, max_tokens)))
    return dict(status='unscored_instrument_smoke', validated_benchmark=False,
                selection_threshold_established=False, cases=results)
