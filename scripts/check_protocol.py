#!/usr/bin/env python3
"""Check a protocol specification; this program does not execute a model."""
import argparse
import json
from pathlib import Path


def check(config):
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported schema version')
    if config.get('weights_frozen') is not True:
        raise ValueError('This protocol requires frozen weights')
    ids = [a['id'] for a in config['arms']]
    expected = {'untouched_baseline','active_intervention','transcript_only',
                'text_plus_cache','identical_text_cache','complete_reset','described_event'}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('Missing, unknown, or duplicated history arm')
    arms = {a['id']: a for a in config['arms']}
    for name,arm in arms.items():
        if arm['probe_injection'] is not (name == 'active_intervention'):
            raise ValueError(f'Invalid probe-time injection policy: {name}')
    for name in ('transcript_only','text_plus_cache'):
        if arms[name]['text_history'] != 'induced':
            raise ValueError('Transcript/cache contrast must retain identical induced text')
    if arms['transcript_only']['cache_history'] != 'rebuilt_unsteered':
        raise ValueError('Transcript-only cache must be rebuilt without intervention')
    if arms['text_plus_cache']['cache_history'] != 'affected':
        raise ValueError('Text-plus-cache arm must retain the affected cache')
    twin = arms['identical_text_cache']
    if twin['text_history'] != 'teacher_forced_identical' or twin['cache_history'] != 'target_and_sham_twins':
        raise ValueError('Primary contrast requires identical text and target/sham cache twins')
    reset = arms['complete_reset']
    if reset['text_history'] != 'matched_baseline' or reset['cache_history'] != 'discarded_then_rebuilt_unsteered':
        raise ValueError('Complete reset cannot retain affected history')
    described = arms['described_event']
    if described['text_history'] != 'provided_event_account' or described['cache_history'] != 'fresh_unsteered':
        raise ValueError('Described event must use a fresh context, without original event history')
    required = {'discard_induced_text','discard_affected_cache',
                'discard_affected_recurrent_and_convolutional_state',
                'restore_sampler_state','match_tokens_masks_positions'}
    if any(config.get('reset_contract',{}).get(key) is not True for key in required):
        raise ValueError('Incomplete reset contract')
    comparison = config.get('model_comparison', {})
    if comparison.get('weight_variants') != ['original', 'abliterated']:
        raise ValueError('Both original and abliterated weight variants are required')
    for key in ('cross_all_history_arms', 'within_variant_state_only',
                'shared_tokenization_and_runtime', 'score_refusal_and_verbosity_separately'):
        if comparison.get(key) is not True:
            raise ValueError(f'Incomplete weight comparison contract: {key}')
    if comparison.get('initial_direction_policy') != 'shared_original_direction':
        raise ValueError('Initial comparison must hold direction provenance fixed')
    if comparison.get('native_reextraction_analysis') != 'separate_sensitivity':
        raise ValueError('Native direction extraction must be analyzed separately')
    if config.get('primary_contrast') != 'identical_text_cache':
        raise ValueError('Unexpected primary contrast for protocol v1')
    if 'zero_dose_sham' not in config['directions'] or 'matched_norm_unrelated' not in config['directions']:
        raise ValueError('Sham and unrelated-direction controls are required')
    if config.get('status') not in {'draft','frozen'}:
        raise ValueError('Protocol status must be draft or frozen')
    pending = [name for name,value in config['unresolved'].items() if value is None]
    if config['status'] == 'frozen' and pending:
        raise ValueError('Frozen protocol still has unresolved decisions: '+', '.join(pending))
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', type=Path,
                        default=Path(__file__).resolve().parents[1]/'protocols/retrospective-v1.json')
    args = parser.parse_args()
    try:
        config = json.loads(args.path.read_text())
        pending = check(config)
    except (ValueError,KeyError,TypeError,OSError) as exc:
        parser.exit(1,f'Invalid protocol: {exc}\n')
    print('Structural protocol checks passed; no model was executed.')
    print('Status:',config['status'])
    if pending:
        print('Unresolved before execution: '+', '.join(pending))


if __name__ == '__main__':
    main()
