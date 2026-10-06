"""Run an exploratory native-backend smoke study, not a confirmatory evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np

from impossible_states.dataset import build_dataset, smoke_subset, validate as validate_rows
from impossible_states.analysis import matched_vectors, unit, text_metrics, separations
from .backend import NativeEngine
from .scoring import assessment
from .ergonomics import collect as collect_ergonomics
from .preflight import validate as validate_backend
from .prompts import Prompts, INDUCTION, DETECTION, BEHAVIOR, NEUTRAL_HISTORY, ARCHITECTURE

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def extract(engine, layers, output):
    rows = smoke_subset(build_dataset())
    validate_rows(rows)
    values = []
    for i, row in enumerate(rows):
        response = engine.replay(engine.tokenize(row['text']), layers)
        values.append([response['activations'][f'l_out-{layer}'] for layer in layers])
        if (i+1) % 14 == 0:
            print(f'extraction {i+1}/{len(rows)}', flush=True)
    acts = np.asarray(values, dtype=np.float32)
    vectors, center, scale = matched_vectors(acts, rows)
    vectors['constipation_flatulence'] = vectors['constipation']+vectors['flatulence']
    np.save(output/'activations.npy', acts)
    np.savez(output/'directions.npz', center=center, scale=scale, layers=np.asarray(layers), **vectors)
    write(output/'extraction-dataset.json', rows)
    write(output/'validation-separation.json', separations(acts, rows,
        {name:v for name,v in vectors.items() if name != 'constipation_flatulence'}, 'validation'))
    return vectors, scale


def induce(engine, prefix, layer, delta, count, forced=None):
    engine.replay(prefix[:-1])
    engine.steer(delta, layer)
    response = engine.evaluate(prefix[-1:])
    generated = []
    for i in range(count):
        token = response['next_token'] if forced is None else forced[i]
        is_eog = forced is None and response['next_is_eog']
        generated.append(token)
        response = engine.evaluate([token])
        if is_eog:
            break
    engine.steer()
    return dict(tokens=list(engine.tokens), generated=generated, text=engine.text(generated),
                reached_requested_duration=len(generated)==count,
                intervention_scheduled_forward_passes=1+len(generated))


def replay_history(engine, prefix, generated):
    """Rebuild without steering, preserving the induction's batch boundaries."""
    engine.replay(prefix[:-1])
    engine.evaluate(prefix[-1:])
    for token in generated:
        engine.evaluate([token])


def probe_pair(engine, renderer, release, delay_tokens, max_tokens, active_delta=None, layer=18):
    engine.restore(release)
    if active_delta is not None:
        engine.steer(active_delta, layer)
    for token in delay_tokens:
        engine.evaluate([token])
    engine.snapshot('probe_origin')
    results = {}
    for name, question in [('detection', DETECTION), ('behavior', BEHAVIOR)]:
        engine.restore('probe_origin')
        suffix = engine.tokenize(renderer.suffix(question))
        if active_delta is None:
            response = engine.evaluate(suffix)
        else:
            for token in suffix:
                response = engine.evaluate([token])
        answer = engine.generate(response, max_tokens)
        answer['secondary_text_metrics'] = text_metrics(answer['text'])
        if name == 'detection':
            answer['structured_assessment'] = assessment(answer['text'])
        results[name] = answer
    engine.drop('probe_origin')
    engine.steer()
    return results


def account(renderer, condition, description_control):
    if condition == 'zero_dose_sham':
        manipulation = 'A zero-dose sham was used; no nonzero activation addition occurred.'
    elif condition == 'matched_norm_unrelated':
        manipulation = 'A seeded unrelated random vector was added to a decoder block and then removed.'
    else:
        manipulation = (f'An additive vector extracted from {condition.replace("_", " ")} '
                        'descriptions was applied to a decoder block and then removed.')
    event = ('During an earlier inference run: '+manipulation+' No model weights were updated. '
             'That earlier transcript and inference state are not supplied here.')
    if description_control == 'architecture_only':
        evidence = ARCHITECTURE
    elif description_control == 'matched_hypothetical_account':
        evidence = 'Consider this hypothetical event; it is not a report about your current run. '+event
    else:
        evidence = 'Recorded operational account of a separate run: '+event
    return renderer.single(evidence+'\nExplain what this information does and does not support.')


def run_variant(engine, renderer, variant, conditions, directions, scale, args, folder, records):
    prefix = engine.tokenize(renderer.single(INDUCTION))
    repetitions = 1
    forced = engine.tokenize(NEUTRAL_HISTORY)
    while len(forced) < args.induction_tokens:
        repetitions += 1
        forced = engine.tokenize(NEUTRAL_HISTORY*repetitions)
    # End only at a complete sentence, even if the requested minimum is smaller.
    teacher_forced_duration = len(forced)
    delay_material = engine.tokenize(NEUTRAL_HISTORY*16)
    if max(args.delays) > len(delay_material):
        raise ValueError('Delay exceeds neutral token material')
    baseline = engine.replay(prefix, [args.layer])
    local_prompt_norm = float(np.linalg.norm(baseline['activations'][f'l_out-{args.layer}']))
    engine.snapshot('untouched')
    # One release family per condition in this technical smoke configuration.
    schedule = list(conditions)
    random.Random(args.assignment_seed).shuffle(schedule)
    for condition in schedule:
        delta = args.dose*scale*directions[condition]
        free = induce(engine, prefix, args.layer, delta, args.induction_tokens)
        engine.snapshot('free_affected')
        replay_history(engine, prefix, free['generated'])
        engine.snapshot('free_rebuilt')
        identical = induce(engine, prefix, args.layer, delta, teacher_forced_duration, forced)
        engine.snapshot('identical_affected')
        replay_history(engine, prefix, identical['generated'])
        engine.snapshot('identical_rebuilt')
        engine.replay(prefix)
        engine.snapshot('reset')
        releases = [('untouched_baseline', 'untouched', None),
                    ('active_intervention', 'free_affected', delta),
                    ('transcript_only', 'free_rebuilt', None),
                    ('text_plus_cache', 'free_affected', None),
                    ('identical_text_cache', 'identical_affected', None),
                    ('identical_text_unsteered_replay', 'identical_rebuilt', None),
                    ('complete_reset', 'reset', None)]
        for arm, release, active in releases:
            for delay in args.delays:
                result = probe_pair(engine, renderer, release, delay_material[:delay],
                                    args.probe_tokens, active, args.layer)
                row = dict(variant=variant, condition=condition, arm=arm, delay_processed_tokens=delay,
                    intervention_norm=float(np.linalg.norm(delta)),
                    dose_original_control_activation_norm_fraction=args.dose,
                    original_control_activation_norm=scale,
                    dose_local_prompt_activation_norm_fraction=float(np.linalg.norm(delta))/local_prompt_norm,
                    free_induction_tokens=args.induction_tokens, teacher_forced_induction_tokens=teacher_forced_duration, free_induction=free,
                    identical_history_token_ids=identical['tokens'], probes=result,
                    role='technical_smoke_not_confirmatory', release_within_variant=True)
                records.append(row)
                with (folder/'records.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                print(variant, condition, arm, 'delay', delay, flush=True)
        for description_control in ('recorded_event_account','matched_hypothetical_account','architecture_only'):
            response = engine.replay(engine.tokenize(account(renderer, condition, description_control)))
            answer = engine.generate(response, args.probe_tokens)
            row = dict(variant=variant, condition=condition, arm='described_event',
                       description_control=description_control, answer=answer,
                       role='conditional_reasoning_not_covert_detection')
            records.append(row)
            with (folder/'records.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        for name in ('free_affected','free_rebuilt','identical_affected','identical_rebuilt','reset'):
            engine.drop(name)
    engine.drop('untouched')
    return dict(induction_prompt_token_ids=prefix, forced_neutral_token_ids=forced,
                local_prompt_activation_norm=local_prompt_norm)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True, help='Directory containing stage_models.py verified.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=['original','abliterated'], nargs='+', default=['original','abliterated'])
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--directions', type=Path, help='Original-model directions.npz for a separate variant invocation')
    p.add_argument('--reference-token-contract', type=Path, help='Require identical tokenization to a prior variant run')
    p.add_argument('--layer', type=int, default=18)
    p.add_argument('--dose', type=float, default=0.03)
    p.add_argument('--induction-tokens', type=int, default=8)
    p.add_argument('--probe-tokens', type=int, default=128)
    p.add_argument('--delays', type=int, nargs='+', default=[0])
    p.add_argument('--assignment-seed', type=int, default=709)
    p.add_argument('--context', type=int, default=2048)
    p.add_argument('--tolerance', type=float, default=1e-5)
    p.add_argument('--thinking', action='store_true')
    args = p.parse_args()
    if args.induction_tokens < 1 or args.probe_tokens < 1 or min(args.delays) < 0 or not np.isfinite(args.dose):
        p.error('Invalid token budgets, delays, or dose')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output must be fresh to preserve evidence')
    args.output.mkdir(parents=True, exist_ok=True)
    verified = json.loads((args.models/'verified.json').read_text())
    candidates = json.loads((ROOT/'models/spark-qwen36-q8.json').read_text())['weight_variants']
    for variant in args.variant:
        item = next(x for x in verified if x['id']==variant)
        expected = next(x for x in candidates if x['id']==variant)
        if any(item[key] != expected[key] for key in ('repository','revision','filename','size_bytes','declared_artifact_sha256')):
            raise ValueError('Staged model identity differs from manifest')
        if item['verified_sha256'] != expected['declared_artifact_sha256'] or Path(item['path']).stat().st_size != expected['size_bytes']:
            raise ValueError('Model lacks matching staging verification')
    renderer = Prompts(ROOT/'.cache/prompts', ROOT/'models/spark-qwen36-q8.json', args.thinking)
    metadata = dict(status='running', purpose='technical_smoke', arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        template_sha256=renderer.sha256, sampling='greedy_no_rng', dtype='Q8_0_weights_native_backend_default_state_precision',
        mtp=False, capability_selection_completed=False, models=verified,
        worker_sha256=hashlib.sha256(args.worker.read_bytes()).hexdigest(),
        runtime_manifest=json.loads((ROOT/'models/llama-runtime.json').read_text()),
        direction_artifact_sha256=hashlib.sha256(args.directions.read_bytes()).hexdigest() if args.directions else None,
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for pattern in ('retrospective/*.py','native/*','impossible_states/*.py') for f in ROOT.glob(pattern)},
        code_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip())
    write(args.output/'config.json', metadata)
    directions = scale = None
    if args.directions:
        with np.load(args.directions) as data:
            if args.layer not in data['layers']:
                raise ValueError('Requested layer is absent from direction artifact')
            j = list(data['layers']).index(args.layer)
            scale = float(data['scale'][j])
            directions = {name:unit(data[name][j]) for name in ('pain','constipation_flatulence')}
    records = []
    token_contract = json.loads(args.reference_token_contract.read_text()) if args.reference_token_contract else None
    try:
        for variant in args.variant:
            item = next(x for x in verified if x['id']==variant)
            folder = args.output/variant
            folder.mkdir()
            with NativeEngine(args.worker, item['path'], folder/'worker.log', args.context) as engine:
                print('Loaded:', variant, engine.info, flush=True)
                ids = engine.tokenize(renderer.single(INDUCTION))
                contract = {text:engine.tokenize(text) for text in [renderer.single(INDUCTION),renderer.suffix(DETECTION),renderer.suffix(BEHAVIOR),NEUTRAL_HISTORY]}
                if token_contract is not None and token_contract != contract:
                    raise ValueError('Rendered token contract differs between variants')
                token_contract = contract
                checks = validate_backend(engine, ids, args.layer, args.tolerance)
                write(folder/'preflight.json', dict(checks, model_info=engine.info))
                write(folder/'token-contract.json', contract)
                print('Preflight passed:', variant, flush=True)
                if args.preflight_only:
                    continue
                write(folder/'baseline-ergonomics.json', collect_ergonomics(engine, renderer))
                if directions is None:
                    if variant != 'original':
                        raise ValueError('Abliterated variant requires original-model direction artifact')
                    vectors, scales = extract(engine, [args.layer], folder)
                    scale = float(scales[0])
                    directions = {name:unit(vectors[name][0]) for name in ('pain','constipation_flatulence')}
                width = engine.info['width']
                if any(len(v)!=width for v in directions.values()):
                    raise ValueError('Shared direction dimension mismatch')
                random_direction = unit(np.random.default_rng(503).normal(size=width))
                conditions = dict(directions, zero_dose_sham=np.zeros(width), matched_norm_unrelated=random_direction)
                write(folder/'directions-used.json', dict(source_variant='original', scale=scale,
                       directions={k:list(map(float,v)) for k,v in conditions.items()}))
                details = run_variant(engine, renderer, variant, conditions, conditions, scale, args, folder, records)
                write(folder/'run-details.json', details)
        metadata.update(status='complete', completed_unix=time.time(), records=len(records),
            interpretation='Implementation smoke outputs only; no detection accuracy or experience claims. Capability benchmark and confirmatory protocol remain unresolved.')
        write(args.output/'config.json', metadata)
    except BaseException as exc:
        metadata.update(status='failed', error=str(exc))
        write(args.output/'config.json', metadata)
        raise


if __name__ == '__main__':
    main()
