#!/usr/bin/env python3
"""Inspect tensor precision and tokenizer metadata without reading tensor weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0,str(args.llama_source/'gguf-py'))
    from gguf import GGUFReader
    models = json.loads((args.models/'verified.json').read_text())
    results = []
    for model in models:
        reader = GGUFReader(model['path'])
        fields = {name:f.contents() for name,f in reader.fields.items()
                  if name.startswith('tokenizer.') or name.startswith('qwen35moe.') or name.startswith('general.')}
        tokenizer = {name:value for name,value in fields.items() if name.startswith('tokenizer.') and 'chat_template' not in name}
        tensors = [{'name':t.name,'type':t.tensor_type.name,'shape':t.shape.tolist()} for t in reader.tensors]
        results.append(dict(variant=model['id'], artifact_sha256=model['verified_sha256'],
            architecture=fields.get('general.architecture'),file_type=fields.get('general.file_type'),
            rope_sections=fields.get('qwen35moe.rope.dimension_sections'),
            metadata_tokenizer_sha256=digest(tokenizer),
            embedded_template_sha256=digest(fields.get('tokenizer.chat_template')),
            tensor_schema_sha256=digest(tensors),tensor_types=dict(Counter(t['type'] for t in tensors)),
            tensors=tensors))
    comparison = None
    if {x['variant'] for x in results}=={'original','abliterated'}:
        a,b=results
        comparison = dict(tokenizer_metadata_equal=a['metadata_tokenizer_sha256']==b['metadata_tokenizer_sha256'],
            tensor_schema_and_precision_equal=a['tensor_schema_sha256']==b['tensor_schema_sha256'],
            embedded_templates_equal=a['embedded_template_sha256']==b['embedded_template_sha256'],
            scope='Artifact compatibility; neither converter recipe equality nor pure abliteration causality is established.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(artifacts=results,comparison=comparison),indent=2)+'\n')
    print(json.dumps({'variants':[x['variant'] for x in results],'comparison':comparison}))


if __name__ == '__main__':
    main()
