import json,pathlib,sys,time,urllib.request
sys.path.insert(0,'/home/shiba/multikernel-experiment/model-api');from validate_tts_api import request
root=pathlib.Path('/home/shiba/multikernel-experiment/deepseek-v41');key=pathlib.Path('/home/shiba/multikernel-experiment/model-api/api-keys/user-2.key').read_text().strip();base='https://shiba2th-1.tail0e03fa.ts.net:8443';model='deepseek-v4.1-flash-1017-uncensored-iq2-xs'
rows=[]
# Switching away must terminate the scoped DeepSeek process and return its memory.
for mid in ['qwen3.8-flash-next-strata',model]:
 start=time.monotonic();r=json.loads(request('/v1/chat/completions',{'model':mid,'messages':[{'role':'user','content':'Reply with a short greeting.'}],'max_tokens':64,'reasoning_effort':'none','temperature':0,'stream':False}));text=r['choices'][0]['message']['content'];assert text;rows.append({'switch_to':mid,'seconds':time.monotonic()-start,'text':text})
headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
with urllib.request.urlopen(urllib.request.Request(base+'/v1/models',headers=headers),timeout=30) as r:models=json.load(r)['data']
assert any(m['id']==model and m['metadata']['active'] for m in models)
body={'model':model,'messages':[{'role':'user','content':'Reply with only the word READY.'}],'max_tokens':32,'temperature':0,'reasoning_effort':'none','stream':True}
req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers=headers);text='';done=False
with urllib.request.urlopen(req,timeout=600) as r:
 for line in r:
  line=line.decode().strip()
  if line=='data: [DONE]':done=True;break
  if line.startswith('data: '):
   event=json.loads(line[6:]);choices=event.get('choices') or []
   if choices:text+=choices[0].get('delta',{}).get('content') or ''
assert done and 'READY' in text,text
out={'switching':rows,'public_model_count':len(models),'public_stream_text':text,'public_stream_done':done};(root/'public-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
