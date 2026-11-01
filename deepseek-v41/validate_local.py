import json,pathlib,re,time,urllib.request,urllib.error
root=pathlib.Path('/home/shiba/multikernel-experiment/deepseek-v41');base='http://127.0.0.1:19085'
started=time.monotonic()
while time.monotonic()-started<3600:
 try:
  with urllib.request.urlopen(base+'/health',timeout=5) as r:
   if json.load(r).get('status')=='ok':break
 except (urllib.error.URLError,TimeoutError):pass
 time.sleep(2)
else:raise RuntimeError('DeepSeek did not become ready')
ready_seconds=time.monotonic()-started
rows=[]
checks=[('capital','What is the capital of France? Reply with only its name.',lambda x:bool(re.search(r'\bParis\b',x,re.I))),('arithmetic','What is 17 plus 25? Reply with only the number.',lambda x:x.strip().rstrip('.')==str(17+25)),('generation','Write a short paragraph explaining how a bicycle works.',lambda x:len(x.split())>=5)]
for label,prompt,check in checks:
 body={'model':'deepseek-v4.1-flash-1017-uncensored-iq2-xs','messages':[{'role':'user','content':prompt}],'max_tokens':64 if label=='generation' else 32,'temperature':0,'top_p':1,'stream':False,'reasoning_effort':'none'}
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'});start=time.monotonic()
 with urllib.request.urlopen(req,timeout=1200) as r:result=json.load(r)
 elapsed=time.monotonic()-start;message=result['choices'][0]['message'];text=message.get('content') or '';passed=check(text)
 rows.append({'case':label,'seconds':elapsed,'passed':passed,'response':result,'completion_tokens_per_wall_second':result.get('usage',{}).get('completion_tokens',0)/elapsed})
 (root/'local-validation.json').write_text(json.dumps({'ready_after_seconds':ready_seconds,'runs':rows},indent=2)+'\n')
 print(label,'PASS' if passed else 'FAIL',round(elapsed,2),'s',repr(text),flush=True)
 assert passed,label+' failed; stopping validation'
print('LOCAL VALIDATION PASSED',flush=True)
