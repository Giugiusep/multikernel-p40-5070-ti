import json, pathlib, sys, time, urllib.request
sys.path.insert(0, '/home/shiba/multikernel-experiment/model-api')
from validate_tts_api import request
ROOT = pathlib.Path(__file__).resolve().parent
MODEL = 'deepseek-v4.1-flash-1017-uncensored-iq2-xs-dspark'
rows = []
for label, prompt in [('prose', 'Explain how a bicycle works in one clear paragraph.'), ('code', 'Write a Python function that returns the first n Fibonacci numbers. Include a brief explanation.')]:
    start = time.monotonic()
    result = json.loads(request('/v1/chat/completions', {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 64, 'temperature': 0, 'top_p': 1, 'top_k': 1, 'seed': 42, 'reasoning_effort': 'none', 'stream': False}))
    assert result['choices'][0]['message']['content']
    rows.append({'case': label, 'wall_seconds': time.monotonic() - start, 'response': result})
    (ROOT / 'api-validation.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(label, result.get('timings'), flush=True)
key = pathlib.Path('/home/shiba/multikernel-experiment/model-api/api-keys/user-2.key').read_text().strip()
body = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Reply with only READY.'}], 'max_tokens': 32, 'temperature': 0, 'reasoning_effort': 'none', 'stream': True}
req = urllib.request.Request('https://shiba2th-1.tail0e03fa.ts.net:8443/v1/chat/completions', data=json.dumps(body).encode(), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
text = ''; done = False
with urllib.request.urlopen(req, timeout=600) as response:
    for line in response:
        line = line.decode().strip()
        if line == 'data: [DONE]': done = True; break
        if line.startswith('data: '):
            event = json.loads(line[6:]); choices = event.get('choices') or []
            if choices: text += choices[0].get('delta', {}).get('content') or ''
assert done and 'READY' in text, text
(ROOT / 'public-validation.json').write_text(json.dumps({'stream_done': done, 'text': text}, indent=2) + '\n')
print('Public second-user streaming passed', flush=True)
