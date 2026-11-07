"""Controlled full DeepSeek decode gate; systemd ExecStopPost restores the API."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
sys.path.insert(0,str(BASE/'model-api'))
from validate_tts_api import request
CFG=json.loads((BASE/'deepseek-v41/strata-deepseek-iq2-xs-dspark.json').read_text())
PROMPTS=json.loads((BASE/'deepseek-v41/optimization/prompts.json').read_text())
REPORT={'contract':'same six experts, DSpark block5/conf0.65, greedy, context8192, readonly placement','runs':[],'passed':False}
def save(): (ROOT/'results.json').write_text(json.dumps(REPORT,indent=2)+'\n')
def guard():
 if subprocess.run(['systemctl','--user','is-active','--quiet','deepseek-decode-release-build-20261107.service']).returncode==0:
  raise RuntimeError('Timed test refused while private build unit is active')
 names=subprocess.check_output(['ps','-eo','comm='],text=True).splitlines()
 found=[v for v in names if v.strip() in {'nvcc','cicc','ptxas','cc1plus','cudafe++'}]
 if found:raise RuntimeError('Timed test refused with compilers running: '+str(found))
def run(tag,exe,prompt='code',profile=False,threads=8):
 guard()
 args=list(CFG['args'])+['--ids',','.join(map(str,PROMPTS[prompt])),'--n','64','--temperature','0','--seed','42']
 env=os.environ.copy();env.update(CFG['env']);env.update({'DS4_STORE_THREADS':str(threads),'DS4_TOKLOG':'1','DS4_ROUND_TIMES':'1'})
 if profile:
  env['DS4_PROF']='1';env['DS4_TRACE']=str(ROOT/(tag+'.routes.bin'))
 else:env.pop('DS4_PROF',None)
 # Keep profile writes disabled, never mutate the production config or executable.
 before=Path('/proc/pressure/io').read_text();start=time.monotonic()
 with (ROOT/(tag+'.stderr')).open('w') as err,(ROOT/(tag+'.tokens')).open('w') as out:
  p=subprocess.run([exe,*args],stdout=out,stderr=err,env=env,timeout=600)
 text=(ROOT/(tag+'.stderr')).read_text();raw=(ROOT/(tag+'.tokens')).read_text().strip().strip(',')
 ids=[int(v) for v in raw.split(',') if v.strip()] if p.returncode==0 else []
 m=re.search(r'decode: (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\)',text)
 row={'tag':tag,'prompt':prompt,'exe':exe,'exe_sha256':hashlib.sha256(Path(exe).read_bytes()).hexdigest(),'profile':profile,'reader_threads':threads,'exit':p.returncode,'total_seconds':time.monotonic()-start,'tokens':ids,'metrics':[l for l in text.splitlines() if any(v in l for v in ['decode:','decode per token:','store:','GPU per token','prof CPU lane','dspark:','token latency','decode reads'])],'io_pressure_before':before,'io_pressure_after':Path('/proc/pressure/io').read_text()}
 if m:row.update(output_tokens=int(m[1]),decode_ms=float(m[2]),reported_tps=float(m[3]))
 previous=next((r for r in REPORT['runs'] if r['prompt']==prompt),None)
 row['exact_agreement']=previous is None or previous['tokens']==ids
 if not row['exact_agreement']:row['first_mismatch']=next((i for i,(a,b) in enumerate(zip(previous['tokens'],ids)) if a!=b),min(len(previous['tokens']),len(ids)))
 REPORT['runs'].append(row);save();print(tag,'exit',p.returncode,'tokens',len(ids),'tps',row.get('reported_tps'),'exact',row['exact_agreement'],flush=True)
 if p.returncode or not ids or not m or not row['exact_agreement']:raise RuntimeError('Correctness/execution gate failed: '+tag)
 return row

def main():
 guard()
 private=str(BASE/'research/deepseek-decode-build/strata-dsv41')
 if not Path(private).is_file():raise RuntimeError('Private engine build incomplete; API untouched')
 health=json.loads(request('/health'));state={'restore_model':health.get('active_model'),'phase':'unloading API'};(ROOT/'state.json').write_text(json.dumps(state)+'\n')
 request('/v1/unload',{});subprocess.run(['systemctl','--user','stop','shiba-model-api.service'],check=True)
 state['phase']='benchmarking';(ROOT/'state.json').write_text(json.dumps(state)+'\n');save()
 reference=CFG['exe']
 run('release-control-a',reference)
 run('release-profile',reference,profile=True)
 run('release-readers16',reference,threads=16)
 run('release-control-b',reference)
 run('private-rebuild',private)
 REPORT['passed']=True;save();state['phase']='complete; restoration pending';(ROOT/'state.json').write_text(json.dumps(state)+'\n')
if __name__=='__main__':main()
