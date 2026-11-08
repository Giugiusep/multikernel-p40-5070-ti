"""Bounded full-engine DeepSeek/P40 integration correctness gate."""
import json,os,re,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
sys.path.insert(0,str(BASE/'model-api'))
from validate_tts_api import request
CFG=json.loads((BASE/'deepseek-v41/strata-deepseek-iq2-xs-dspark.json').read_text())
PROMPTS=json.loads((BASE/'deepseek-v41/optimization/prompts.json').read_text())
EXE=str(BASE/'research/deepseek-decode-build/strata-dsv41')
REPORT={'runs':[],'passed':False,'contract':'CPU expert replacement only, original six experts/Engram/DSpark/cache32/CPU8; private sm120 coordinator and SM61 worker6011 Q8_K,1GiB fixed admission'}
def save():(ROOT/'results.json').write_text(json.dumps(REPORT,indent=2)+'\n')
def run(mode,prompt,n):
 tag=f'{mode}-{prompt}-{n}';env=os.environ.copy();env.update(CFG['env']);env.update(DS4_MK_EXPERT=mode,DS4_TOKLOG='1',DS4_ROUND_TIMES='1')
 args=list(CFG['args'])+['--ids',','.join(map(str,PROMPTS[prompt])),'--n',str(n),'--temperature','0','--seed','42']
 start=time.monotonic()
 with (ROOT/(tag+'.tokens')).open('w') as out,(ROOT/(tag+'.stderr')).open('w') as err:
  p=subprocess.run([EXE,*args],stdout=out,stderr=err,env=env,timeout=900)
 text=(ROOT/(tag+'.stderr')).read_text();raw=(ROOT/(tag+'.tokens')).read_text().strip().strip(',');ids=[int(v) for v in raw.split(',') if v.strip()] if p.returncode==0 else []
 ref=next((v['tokens'] for v in REPORT['runs'] if v['mode']=='off' and v['prompt']==prompt and v['n']==n),None)
 m=re.search(r'decode: (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\)',text)
 errors=[float(v) for v in re.findall(r'mk expert shadow .*?rel_l1=([\d.e+-]+)',text)]
 row={'mode':mode,'prompt':prompt,'n':n,'exit':p.returncode,'tokens':ids,'exact_agreement':ref is None or ref==ids,'wall_s':time.monotonic()-start,'loads':len(re.findall('mk expert load ',text)),'shadow_count':len(errors),'max_shadow_rel_l1':max(errors,default=None),'worker_failed':'mk expert disabled:' in text,'metrics':[l for l in text.splitlines() if any(k in l for k in ['decode:','decode per token:','store:','dspark:'])]}
 if m:row['tps']=float(m[3])
 if ref is not None and ref!=ids:row['first_mismatch']=next((i for i,(a,b) in enumerate(zip(ref,ids)) if a!=b),min(len(ref),len(ids)))
 REPORT['runs'].append(row);save();print(tag,{k:v for k,v in row.items() if k not in ('tokens','metrics')},flush=True)
 if p.returncode or not m or not row['exact_agreement'] or row['worker_failed']:raise RuntimeError('gate failed: '+tag)
 if mode=='shadow' and (not errors or max(errors)>.001):raise RuntimeError('real CPU/P40 numerical gate failed')
def main():
 if subprocess.run(['systemctl','--user','is-active','--quiet','deepseek-q8k-build-20261108.service']).returncode==0:raise RuntimeError('build active')
 REPORT['binary_sha256']=hashlib.sha256(Path(EXE).read_bytes()).hexdigest()
 state={'restore_model':json.loads(request('/health')).get('active_model')};(ROOT/'state.json').write_text(json.dumps(state))
 request('/v1/unload',{});subprocess.run(['systemctl','--user','stop','shiba-model-api.service'],check=True)
 for prompt in ['code','prose']:
  for mode in ['off','shadow','replace']:run(mode,prompt,32)
 REPORT['passed']=True;save()
if __name__=='__main__':main()
