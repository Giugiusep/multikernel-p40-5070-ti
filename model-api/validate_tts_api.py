"""Exercise authenticated speech and confirm it leaves the chat model resident."""
import array
import io
import json
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:19080'
KEY = (ROOT / 'api-key').read_text().strip()

def request(path, payload=None, authorized=True):
    headers = {'Content-Type': 'application/json'}
    if authorized:
        headers['Authorization'] = 'Bearer ' + KEY
    req = urllib.request.Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=600) as response:
        return response.read()

if __name__ == '__main__':
    # Restore the previously active vision profile after the API restart.
    request('/v1/switch', {'model': 'qwen3.8-flash-next-strata-vision'})
    before = json.loads(request('/health'))['active_model']
    results = []
    for i in range(3):
        start = time.monotonic()
        audio = request('/v1/audio/speech', {'model': 'piper-tts', 'voice': 'lessac',
            'input': 'Hello! Your local speech service is ready. I can speak while your language model stays loaded.', 'response_format': 'wav'})
        elapsed = time.monotonic() - start
        with wave.open(io.BytesIO(audio)) as wav:
            assert wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == 22050
            duration = wav.getnframes() / wav.getframerate()
            samples = array.array('h', wav.readframes(wav.getnframes()))
            assert duration > 1 and max(abs(x) for x in samples) > 100
        results.append({'seconds': elapsed, 'audio_seconds': duration, 'real_time_factor': elapsed / duration, 'bytes': len(audio)})
        if i == 0:
            (ROOT / 'tts-sample.wav').write_bytes(audio)
    for payload, auth, expected in [({'input': 'hi'}, False, 401), ({'input': ''}, True, 400),
                                    ({'input': 'hi', 'response_format': 'mp3'}, True, 400),
                                    ({'input': 'hi', 'speed': 0}, True, 400)]:
        try:
            request('/v1/audio/speech', payload, auth)
            raise AssertionError('expected rejection')
        except urllib.error.HTTPError as exc:
            assert exc.code == expected
    assert json.loads(request('/health'))['active_model'] == before
    output = {'voice': json.loads(request('/v1/audio/voices')), 'runs': results,
              'active_model_preserved': before, 'auth_and_validation': 'passed'}
    (ROOT / 'tts-api-validation.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))
