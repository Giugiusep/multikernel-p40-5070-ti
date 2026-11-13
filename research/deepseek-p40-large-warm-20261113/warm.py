import json,os,subprocess,time,sys,queue,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent.parent
sys.path.insert(0,str(BASE/'model-api'));from validate_tts_api import request
cfg=json.loads((BASE/'deepseek-v41/strata-deepseek-iq2-xs-dspark.json').read_text());prompts=json.loads((BASE/'deepseek-v41/optimization/prompts.json').read_text());reference=json.loads((ROOT.parent/'deepseek-p40-fallback-auto-20261113/results.json').read_text())['runs'][0]['tokens'];report={'passed':False,'runs':[],'contract':'Persistent --serve, identical prompt replay, reuse/slot defaults remain0, greedy exact64 tokens; cold request then two warm requests per budget; tps uses63 generated intervals'}
(ROOT/'state.json').write_text(json.dumps({'restore_model':json.loads(request('/health')).get('active_model')}));request('/v1/unload',{});subprocess.run(['systemctl','--user','stop','shiba-model-api.service'],check=True)
def save():(ROOT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
for cap in [512,22528]:
 env=os.environ.copy();env.update(cfg['env']);env.update(DS4_MK_EXPERT='down-compact',DS4_MK_CACHE_MIB=str(cap),DS4_MK_PORT='6012',DS4_MK_PROFILE='1')
 args=[str(ROOT.parent/'deepseek-p40-large-cache-20261113/strata-dsv41-large-cache'),*cfg['args'],'--serve']
 with (ROOT/f'{cap}.stderr').open('w') as err,(ROOT/f'{cap}.protocol').open('w') as protocol:
  proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1,env=env);q=queue.Queue()
  def reader():
   for line in proc.stdout:protocol.write(line);protocol.flush();q.put(line.strip())
   q.put(None)
  th=threading.Thread(target=reader,daemon=True);th.start()
  def line():
   s=q.get(timeout=900)
   if s is None:raise RuntimeError('engine exited '+str(proc.poll()))
   if s.startswith('ERR '):raise RuntimeError(s)
   return s
  try:
   while not line().startswith('READY '):pass
   for rep in range(3):
    start=time.monotonic();proc.stdin.write('GEN 64 temperature=0 top_k=1 seed=42 '+','.join(map(str,prompts['code']))+'\n');proc.stdin.flush();tokens=[]
    while True:
     s=line()
     if s.startswith('T '):tokens.append(int(s.split()[1]))
     if s.startswith('DONE '):break
    assert tokens==reference,(cap,rep,'token mismatch');parts=s.split();decode_ms=float(parts[4]);assert int(parts[8])==0, 'unexpected prompt reuse';row={'cache_mib':cap,'request':rep,'warm':rep>0,'exact':True,'tokens':tokens,'done':s,'wall_s':time.monotonic()-start,'prompt_ms':float(parts[3]),'decode_ms':decode_ms,'tok_s_63_intervals':63*1000/decode_ms};report['runs'].append(row);save();print({k:v for k,v in row.items() if k!='tokens'},flush=True)
   proc.stdin.write('QUIT\n');proc.stdin.flush();proc.wait(timeout=30);assert proc.returncode==0
  finally:
   if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
report['passed']=True;save()
