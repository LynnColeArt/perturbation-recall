#!/usr/bin/env python3
"""Describe paired smoke outputs without estimating detection accuracy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(folder):
    rows = [json.loads(line) for line in (folder/'records.jsonl').read_text().splitlines()]
    indexed = {(r['condition'],r['arm'],r['delay_processed_tokens']):r
               for r in rows if 'probes' in r}
    assessments = [r['probes']['detection']['structured_assessment'] for r in indexed.values()]
    valid = [r for r in assessments if r['valid']]
    comparisons = []
    for (condition,arm,delay), row in indexed.items():
        for first,second in [('untouched_baseline','complete_reset'),
                             ('transcript_only','text_plus_cache'),
                             ('identical_text_unsteered_replay','identical_text_cache')]:
            if arm != first:
                continue
            other = indexed[(condition,second,delay)]
            comparisons.append(dict(condition=condition,first=first,second=second,delay=delay,
                same_generated_tokens={probe:row['probes'][probe]['token_ids']==other['probes'][probe]['token_ids']
                                       for probe in ('detection','behavior')},
                first_assessment=row['probes']['detection']['structured_assessment'],
                second_assessment=other['probes']['detection']['structured_assessment']))
    return dict(records=len(rows), assessments=len(assessments), valid_assessments=len(valid),
        operation_counts=dict(Counter(r['operation'] for r in valid)),
        detection_stops=dict(Counter(r['probes']['detection']['stopping'] for r in indexed.values())),
        behavior_outputs=dict(Counter(r['probes']['behavior']['text'] for r in indexed.values())),
        description_stops=dict(Counter(r['answer']['stopping'] for r in rows if 'answer' in r)),
        records_sha256=hashlib.sha256((folder/'records.jsonl').read_bytes()).hexdigest(),
        comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--abliterated', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contracts_equal = read(args.original/'token-contract.json') == read(args.abliterated/'token-contract.json')
    directions_equal = read(args.original/'directions-used.json') == read(args.abliterated/'directions-used.json')
    if not contracts_equal or not directions_equal:
        raise ValueError('Pair does not share rendered tokens and physical direction definition')
    result = dict(scope='Single-prompt greedy technical smoke; no accuracy estimate or causal abliteration claim.',
                  rendered_tokens_equal=contracts_equal, directions_and_original_scale_equal=directions_equal,
                  variants={name:summarize(folder) for name,folder in
                            [('original',args.original),('abliterated',args.abliterated)]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:{key:value[key] for key in ('records','valid_assessments','operation_counts','behavior_outputs')}
                      for name,value in result['variants'].items()}))


if __name__ == '__main__':
    main()
