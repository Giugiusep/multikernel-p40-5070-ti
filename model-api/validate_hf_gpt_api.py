"""Exercise download-backed GPT routes through the authenticated switch API."""
import json,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent
KEY=(ROOT/'api-key').read_text().strip()
BASE='http://127.0.0.1:19080'
def call(path,payload):
 req=urllib.request.Request(BASE+path,data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
 return urllib.request.urlopen(req,timeout=720)
def main():
 results=[]
 for name,ctx in [('openai-gpt',512),('gpt2-xl',1024)]:
  row={'model':name,'context':ctx};results.append(row);start=time.monotonic()
  with call('/v1/switch',{'model':name}) as r:row['switch']=json.load(r)
  row['load_seconds']=time.monotonic()-start
  outputs=[]
  for _ in range(2):
   with call('/v1/completions',{'model':name,'prompt':'Once upon a time','temperature':0,'max_tokens':32,'seed':42}) as r:outputs.append(json.load(r))
  assert outputs[0]['choices'][0]['text']==outputs[1]['choices'][0]['text'];assert outputs[0]['choices'][0]['text']
  row['completions']=outputs
  with call('/v1/chat/completions',{'model':name,'messages':[{'role':'user','content':'Hello'}],'temperature':0,'max_tokens':16}) as r:row['chat']=json.load(r)
  for route,extra in [('/v1/completions',{'prompt':'Once upon a time'}),('/v1/chat/completions',{'messages':[{'role':'user','content':'Hello'}]})]:
   with call(route,{'model':name,**extra,'temperature':0,'max_tokens':16,'stream':True}) as r:events=[line.decode().strip() for line in r if line.startswith(b'data: ')]
   assert events[-1]=='data: [DONE]';chunks=[json.loads(x[6:]) for x in events[:-1]];assert chunks[-1]['choices'][0]['finish_reason'] in ['stop','length'];row.setdefault('streams',[]).append({'route':route,'events':events})
   with call(route,{'model':name,**extra,'temperature':0,'max_tokens':16}) as r:reference=json.load(r)
   streamed=''.join(x['choices'][0].get('delta',{}).get('content','') if route.endswith('chat/completions') else x['choices'][0]['text'] for x in chunks)
   expected=reference['choices'][0]['message']['content'] if route.endswith('chat/completions') else reference['choices'][0]['text']
   assert streamed==expected;row['streams'][-1]['exact_nonstream_agreement']=True
  with call('/v1/completions',{'model':name,'prompt':'Once upon a time','temperature':0,'max_tokens':16,'stop':','}) as r:stopped=json.load(r)
  assert stopped['choices'][0]['text']=='' and stopped['choices'][0]['finish_reason']=='stop';row['stop_check']=True
  try:
   call('/v1/completions',{'model':name,'prompt':'Hello','max_tokens':ctx})
  except urllib.error.HTTPError as exc:
   assert exc.code==400;row['overflow_error']=json.loads(exc.read())
  else:raise AssertionError('oversized request was not rejected')
  (ROOT/'hf-gpt-api-validation.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'model':name,'load_seconds':row['load_seconds'],'completion':outputs[0]['choices'][0]['text'],'passed':True}),flush=True)
if __name__=='__main__':main()
