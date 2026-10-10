"""Validate single-request Strata text/image input to text plus spoken WAV output."""
import base64
import io
import json
import sys
import time
import urllib.error
import wave
from pathlib import Path
from validate_tts_api import request

ROOT = Path(__file__).resolve().parent
MODEL = 'qwen3.8-flash-next-strata-vision'
if __name__ == '__main__':
    for attempt in range(30):
        try:
            request('/health')
            break
        except urllib.error.URLError:
            time.sleep(1)
    base = {'model':MODEL,'messages':[{'role':'user','content':'Reply with exactly: Hello friend.'}],
            'max_tokens':96,'temperature':0,'stream':False}
    plain = json.loads(request('/v1/chat/completions',base))['choices'][0]['message']
    payload = dict(base,modalities=['text','audio'],audio={'voice':'lessac','format':'wav'})
    results=[]
    for label, body in [('text',payload),('vision',dict(payload,messages=[{'role':'user','content':[
        {'type':'text','text':'What colour is this image? Reply with only the colour.'},
        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode((ROOT/'vision-fixtures/blue.png').read_bytes()).decode()}}]}]))]:
        started=time.monotonic();result=json.loads(request('/v1/chat/completions',body));message=result['choices'][0]['message']
        assert message['audio']['transcript']==message['content']
        if label=='text':assert message['content']==plain['content']
        else:assert 'blue' in message['content'].lower()
        audio=base64.b64decode(message['audio']['data'],validate=True)
        with wave.open(io.BytesIO(audio)) as wav:
            assert wav.getframerate()==22050 and wav.getnchannels()==1 and wav.getnframes()>1000
            seconds=wav.getnframes()/wav.getframerate()
        (ROOT/f'strata-{label}-speech.wav').write_bytes(audio)
        results.append({'case':label,'text':message['content'],'audio_bytes':len(audio),
                        'audio_seconds':seconds,'request_seconds':time.monotonic()-started})
    try:
        request('/v1/chat/completions',dict(payload,stream=True))
        raise AssertionError('streaming audio should be rejected')
    except urllib.error.HTTPError as exc:assert exc.code==400
    assert json.loads(request('/health'))['active_model']==MODEL
    (ROOT/'strata-speech-validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
