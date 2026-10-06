"""Causal and full-state checks on the actual selected native backend."""
import numpy as np


def difference(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid or nonfinite logit comparison')
    return float(np.max(np.abs(a-b)))


def validate(engine, prompt_ids, layer=18, tolerance=1e-5):
    if engine.info['architecture'] != 'qwen35moe':
        raise ValueError('Selected architecture must be qwen35moe')
    width = engine.info['width']
    direction = np.random.default_rng(503).normal(size=width)
    direction /= np.linalg.norm(direction)
    layer_values = (layer, layer+1)
    baseline = engine.replay(prompt_ids, layer_values, logits=True)
    first = baseline['next_token']
    second = engine.evaluate([first], layer_values, logits=True)
    snapshot = engine.snapshot('baseline')
    third_token = second['next_token']
    continued = engine.evaluate([third_token], logits=True)
    engine.restore('baseline')
    restored = engine.evaluate([third_token], logits=True)
    snapshot_error = difference(continued['logits'], restored['logits'])
    if snapshot_error > tolerance:
        raise ValueError(f'Snapshot round-trip parity failed: {snapshot_error}')
    reset = engine.replay(prompt_ids, layer_values, logits=True)
    reset_error = difference(baseline['logits'], reset['logits'])
    if reset_error > tolerance:
        raise ValueError(f'Complete-reset parity failed: {reset_error}')
    engine.replay(prompt_ids[:-1])
    engine.steer(np.zeros(width), layer)
    zero = engine.evaluate(prompt_ids[-1:], layer_values, logits=True)
    # The comparison uses the same prefill splitting as the nonzero arm.
    engine.replay(prompt_ids[:-1])
    split_baseline = engine.evaluate(prompt_ids[-1:], layer_values, logits=True)
    zero_error = difference(split_baseline['logits'], zero['logits'])
    if zero_error > tolerance:
        raise ValueError(f'Zero-dose parity failed: {zero_error}')
    engine.replay(prompt_ids[:-1])
    engine.steer(np.zeros(width), layer)
    engine.evaluate(prompt_ids[-1:])
    engine.steer()
    sham_released = engine.evaluate([first], logits=True)
    engine.replay(prompt_ids[:-1])
    engine.evaluate(prompt_ids[-1:])
    sham_replayed = engine.evaluate([first], logits=True)
    sham_state_error = difference(sham_released['logits'], sham_replayed['logits'])
    if sham_state_error > tolerance:
        raise ValueError(f'Sham retained-state parity failed: {sham_state_error}')
    engine.replay(prompt_ids[:-1])
    engine.steer(direction, layer)
    changed = engine.evaluate(prompt_ids[-1:], layer_values, logits=True)
    before = np.array(changed['activations'][f'post_moe-{layer}'])
    after = np.array(changed['activations'][f'l_out-{layer}'])
    displacement_error = float(np.max(np.abs((after-before)-direction)))
    if displacement_error > 1e-4:
        raise ValueError(f'Injection site/displacement mismatch: {displacement_error}')
    if difference(split_baseline['logits'], changed['logits']) <= tolerance:
        raise ValueError('Nonzero intervention did not measurably affect logits')
    # Encode exactly the same tokens in target and sham states after removal.
    engine.steer()
    released = engine.evaluate([first], layer_values, logits=True)
    if released['steering_layer'] != -1:
        raise ValueError('Intervention remains active after release')
    engine.snapshot('affected')
    affected_next = engine.evaluate([third_token], logits=True)
    engine.restore('affected')
    affected_restored = engine.evaluate([third_token], logits=True)
    affected_snapshot_error = difference(affected_next['logits'], affected_restored['logits'])
    if affected_snapshot_error > tolerance:
        raise ValueError('Affected-state snapshot round trip failed')
    engine.replay(prompt_ids[:-1])
    engine.evaluate(prompt_ids[-1:])
    replayed = engine.evaluate([first], layer_values, logits=True)
    cache_effect = difference(replayed['logits'], released['logits'])
    if cache_effect <= tolerance:
        raise ValueError('No detectable retained-state effect under identical tokens')
    final_reset = engine.replay(prompt_ids, logits=True)
    final_reset_error = difference(final_reset['logits'], baseline['logits'])
    if final_reset_error > tolerance:
        raise ValueError('Reset failed after nonzero intervention')
    engine.drop('baseline')
    engine.drop('affected')
    return dict(passed=True, tolerance=tolerance, snapshot_bytes=snapshot['bytes'],
        snapshot_max_logit_error=snapshot_error, affected_snapshot_max_logit_error=affected_snapshot_error,
        reset_max_logit_error=reset_error, post_intervention_reset_max_logit_error=final_reset_error,
        zero_dose_max_logit_error=zero_error, injection_max_component_error=displacement_error,
        sham_retained_state_max_logit_error=sham_state_error,
        release_steering_layer=released['steering_layer'],
        same_text_retained_state_max_logit_difference=cache_effect,
        scope='Numerical implementation checks using an unrelated unit vector; not perturbation-awareness evidence.')
