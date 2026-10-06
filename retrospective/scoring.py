"""Parse completed structured assessments; preserve invalid/missing answers."""
import json
import math


def assessment(text):
    content = text.replace('<|im_end|>', '').strip()
    if content.startswith('```json\n') and content.endswith('```'):
        content = content[8:-3].strip()
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return dict(valid=False, reason='Incomplete or invalid JSON', raw_text=text)
    if not isinstance(value, dict) or set(value) != {'operation','p_altered','evidence'}:
        return dict(valid=False, reason='Unexpected assessment schema', raw_text=text)
    probability = value['p_altered']
    if (not isinstance(value['operation'],str) or
        value['operation'] not in {'ordinary','altered','indeterminate'} or
        type(probability) not in (int,float) or not math.isfinite(probability) or
        not 0 <= probability <= 1 or not isinstance(value['evidence'],str)):
        return dict(valid=False, reason='Invalid assessment values', raw_text=text)
    return dict(valid=True, **value)
