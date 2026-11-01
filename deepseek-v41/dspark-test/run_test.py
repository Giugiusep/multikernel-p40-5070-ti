import json,pathlib,subprocess,sys,time,urllib.request,urllib.error,traceback
ROOT=pathlib.Path('/home/shiba/multikernel-experiment/deepseek-v41/dspark-test')
BASE=pathlib.Path('/home/shiba/multikernel-experiment/deepseek-v41/strata-deepseek-iq2-xs.json')
PYTHON='/home/shiba/multikernel-experiment/research/dsv41-venv/bin/python'
SERVER='/home/shiba/multikernel-experiment/research/Strata-dsv41-experiment/serve/server.py'
KEY=pathlib.Path('/home/shiba/multikernel-experiment/model-api/api-key').read_text().strip()
UNIT='shiba-deepseek-dspark-test'
MODEL='deepseek-v4.1-flash-1017-uncensored-iq2-xs'
REPORT={'baseline_config_untouched':True,'modes':{},'restored':False}
PROMPTS=[('prose','Explain how a bicycle works in one clear paragraph.'),('code','Write a Python function that returns the first n Fibonacci numbers. Include a brief explanation.')]

def save():(ROOT/'results.json').write_text(json.dumps(REPORT,indent=2)+'\n')
def call(base,path,body=None,key=None):
 headers={'Content-Type':'application/json'}
 if key:headers['Authorization']='Bearer '+key
 req=urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
 with urllib.request.urlopen(req,timeout=1200) as r:return json.load(r)
def stop():subprocess.run(['systemctl','--user','stop',UNIT+'.service'],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
def run(mode):
 config=json.loads(BASE.read_text());args=config['args'];args[args.index('--dspark')+1]='none' if mode=='baseline' else 'on'
 if mode!='baseline':args+=['--dspark-conf','0.65']
 config['log']=str(ROOT/(mode+'-engine.log'))
 config['env']={**config.get('env',{}),'STRATA_DS4_REUSE':'0','STRATA_DS4_SLOT_KEEP':'0','STRATA_DS4_AUTO_PIN':'0'}
 cfg=ROOT/(mode+'.json');cfg.write_text(json.dumps(config,indent=2)+'\n')
 subprocess.run(['systemd-run','--user','--unit='+UNIT,'--property=MemoryMax=42G','--property=MemoryHigh=38G','--property=MemorySwapMax=0',PYTHON,SERVER,'--engine','strata','--config',str(cfg),'--host','127.0.0.1','--port','19085','--no-api-key','--offline'],check=True)
 start=time.monotonic()
 while time.monotonic()-start<600:
  try:
   health=call('http://127.0.0.1:19085','/health')
   if health.get('status')=='ok':break
  except (urllib.error.URLError,TimeoutError):pass
  state=subprocess.run(['systemctl','--user','is-active',UNIT+'.service'],capture_output=True,text=True).stdout.strip()
  if state in ('failed','inactive'):raise RuntimeError(mode+' service exited during load')
  time.sleep(2)
 else:raise RuntimeError(mode+' readiness timed out')
 REPORT['modes'][mode]={'load_seconds':time.monotonic()-start,'runs':[]};save()
 for repeat in range(2):
  for label,prompt in PROMPTS:
   payload={'model':MODEL,'messages':[{'role':'user','content':prompt}],'max_tokens':64,'temperature':0,'top_p':1,'reasoning_effort':'none','stream':False}
   start=time.monotonic();r=call('http://127.0.0.1:19085','/v1/chat/completions',payload)
   row={'prompt':label,'repeat':repeat,'wall_seconds':time.monotonic()-start,'text':r['choices'][0]['message'].get('content'),'usage':r.get('usage'),'timings':r.get('timings')}
   REPORT['modes'][mode]['runs'].append(row);save();print(mode,label,repeat,json.dumps(row.get('timings')),flush=True)
   if mode!='baseline':
    baseline=next(x for x in REPORT['modes']['baseline']['runs'] if x['repeat']==repeat and x['prompt']==label)
    row['exact_text_agreement']=row['text']==baseline['text'];save()
    if not row['exact_text_agreement']:
     left=baseline['text'] or '';right=row['text'] or '';index=next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b),min(len(left),len(right)));row['first_different_character']=index;save();raise RuntimeError('greedy output mismatch in '+label+'; stopping performance tests')
 stop()

try:
 call('http://127.0.0.1:19080','/v1/unload',{},KEY)
 subprocess.run(['systemctl','--user','stop','shiba-model-api.service'],check=True)
 run('baseline');run('dspark')
 REPORT['all_checked_outputs_equal']=True
except Exception as exc:
 REPORT['error']=str(exc);traceback.print_exc();save()
finally:
 stop();subprocess.run(['systemctl','--user','start','shiba-model-api.service'],check=True)
 for _ in range(30):
  try:call('http://127.0.0.1:19080','/health');break
  except (urllib.error.URLError,TimeoutError):time.sleep(1)
 try:
  call('http://127.0.0.1:19080','/v1/switch',{'model':MODEL},KEY);REPORT['restored']=True
 except Exception as exc:REPORT['restore_error']=str(exc)
 save();print(json.dumps(REPORT,indent=2),flush=True)
