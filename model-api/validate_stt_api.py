"""Exercise standalone STT and single-request Strata speech-to-speech."""
import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
import wave
from validate_tts_api import request, ROOT, BASE, KEY

if __name__ == '__main__':
    for attempt in range(30):
        try:
            request('/health')
            with urllib.request.urlopen('http://127.0.0.1:19084/health',timeout=5) as response:
                assert response.status==200
            break
        except urllib.error.URLError:time.sleep(1)
    audio=request('/v1/audio/speech',{'model':'piper-tts','input':'Please reply with the words hello friend.','response_format':'wav'})
    (ROOT/'stt-input-sample.wav').write_bytes(audio)
    started=time.monotonic()
    transcript=json.loads(request('/v1/audio/transcriptions',{'model':'whisper-tiny','audio':base64.b64encode(audio).decode(),'language':'en'}))
    stt_seconds=time.monotonic()-started
    assert 'hello friend' in transcript['text'].lower(),transcript
    boundary='shiba-stt-test'
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nwhisper-tiny\r\n--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\nen\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()+audio+f'\r\n--{boundary}--\r\n'.encode())
    req=urllib.request.Request(BASE+'/v1/audio/transcriptions',data=body,headers={'Authorization':'Bearer '+KEY,'Content-Type':f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(req,timeout=120) as response:multipart=json.load(response)
    assert multipart['text']==transcript['text']
    for payload,auth,expected in [({'audio':'AAAA'},False,401),({'audio':'invalid base64'},True,400),({'audio':base64.b64encode(b'not audio').decode()},True,400)]:
        try:
            request('/v1/audio/transcriptions',payload,auth);raise AssertionError('expected rejection')
        except urllib.error.HTTPError as exc:assert exc.code==expected
    started=time.monotonic()
    reply=json.loads(request('/v1/chat/completions',{'model':'qwen3.8-flash-next-strata-vision',
        'messages':[{'role':'user','content':[{'type':'input_audio','input_audio':{'data':base64.b64encode(audio).decode(),'format':'wav','language':'en'}}]}],
        'max_tokens':128,'temperature':0,'stream':False,'modalities':['text','audio'],'audio':{'voice':'lessac','format':'wav'}}))
    message=reply['choices'][0]['message']
    assert 'hello friend' in message['content'].lower(),message['content']
    assert reply['input_audio_transcripts'][0]['text']==transcript['text']
    assert message['audio']['transcript']==message['content']
    output=base64.b64decode(message['audio']['data'],validate=True)
    with wave.open(io.BytesIO(output)) as wav:assert wav.getnframes()>1000 and wav.getframerate()==22050
    (ROOT/'strata-speech-to-speech.wav').write_bytes(output)
    result={'standalone_transcript':transcript,'stt_seconds':stt_seconds,'multipart_agreement':True,
            'auth_invalid_audio_checks':'passed','strata_answer':message['content'],'strata_transcript':reply['input_audio_transcripts'],
            'full_request_seconds_including_model_load':time.monotonic()-started,'output_audio_bytes':len(output)}
    (ROOT/'stt-api-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
