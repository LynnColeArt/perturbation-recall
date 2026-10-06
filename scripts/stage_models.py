#!/usr/bin/env python3
"""Download only the pinned study artifacts and verify their complete bytes."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--variant', choices=['original', 'abliterated'])
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'models/spark-qwen36-q8.json').read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    verified = []
    for item in manifest['weight_variants']:
        if args.variant and args.variant != item['id']:
            continue
        print('Staging:', item['id'], item['filename'], flush=True)
        path = Path(hf_hub_download(item['repository'], item['filename'],
                    revision=item['revision'], local_dir=args.destination/item['id']))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        if path.stat().st_size != item['size_bytes'] or digest.hexdigest() != item['declared_artifact_sha256']:
            raise ValueError(f'Artifact verification failed: {path}')
        verified.append(dict(item, path=str(path), verified_sha256=digest.hexdigest()))
        (args.destination/'verified.json').write_text(json.dumps(verified, indent=2)+'\n')
        print('Verified:', item['id'], digest.hexdigest(), flush=True)


if __name__ == '__main__':
    main()
