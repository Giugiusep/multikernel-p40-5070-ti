"""RAM cache budget ABBA gate, exact tokens and measured backing/pressure."""
import hashlib,json,os
from pathlib import Path
import re,resource,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
BASE=Path('/home/shiba/multikernel-experiment')
sys.path.insert(0,str(BASE/'model-api'))
from validate_tts_api import request
CFG=json.loads((BASE/'deepseek-v41/strata-deepseek-iq2-xs-dspark.json').read_text())
PROMPTS=json.loads((BASE/'deepseek-v41/optimization/prompts.json').read_text())
REFERENCE={r['prompt']:r['tokens'] for r in json.loads((ROOT.parent/'depth-sweep/results.json').read_text())['runs'] if r['verification_draft_cap']==5}
EXE=CFG['exe']
LONG_REFERENCE={}
REPORT={'runs':[],'passed':False,'contract':'same deployed binary; only expert RAM cache32GB versus38GB changes; soft limit40GiB in both; hard limit42GiB'}
def write(): (ROOT/'results.json').write_text(json.dumps(REPORT,indent=2)+'\n')
def guard():
 for unit in ['deepseek-store-thp-build-20261107.service','deepseek-decode-release-build-20261107.service']:
  if subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0:raise RuntimeError('Build still active')
 names=subprocess.check_output(['ps','-eo','comm='],text=True).splitlines()
 if any(n.strip() in {'cicc','nvcc','ptxas','cc1plus'} for n in names):raise RuntimeError('Compiler visible')
def pressure():return {v:Path('/proc/pressure/'+v).read_text() for v in ['memory','io','cpu']}
def cgstats():
 result={}
 try:
  group=next(l.split(':',2)[2] for l in Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0::'))
  directory=Path('/sys/fs/cgroup')/group.lstrip('/')
  for name in ['memory.current','memory.peak','memory.events','memory.stat','memory.pressure','cpu.stat']:
   result[name]=(directory/name).read_text()
 except OSError as e:result['error']=str(e)
 return result

def run(tag,prompt,cachegb,tokens=64):
 guard();env=os.environ.copy();env.update(CFG['env']);env.update({'DS4_STORE_THP':'0','DS4_TOKLOG':'1','DS4_ROUND_TIMES':'1','DS4_GG_VERBOSE':'1'})
 baseargs=list(CFG['args']);baseargs[baseargs.index('--ssd-cache')+1]=str(cachegb)
 args=baseargs+['--ids',','.join(map(str,PROMPTS[prompt])),'--n',str(tokens),'--temperature','0','--seed','42']
 outpath=ROOT/(tag+'.tokens');errpath=ROOT/(tag+'.stderr');before=pressure();cgbefore=cgstats();r0=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();snapshot=None
 with outpath.open('w') as out,errpath.open('w') as err:
  p=subprocess.Popen([EXE,*args],stdout=out,stderr=err,env=env)
  try:
   while p.poll() is None:
    if snapshot is None and outpath.stat().st_size>0:
     t=time.perf_counter()
     try:snapshot={'smaps_rollup':Path(f'/proc/{p.pid}/smaps_rollup').read_text(),'status':Path(f'/proc/{p.pid}/status').read_text(),'cgroup':cgstats()}
     except OSError as e:snapshot={'error':str(e)}
     snapshot['read_ms']=(time.perf_counter()-t)*1000
    if time.monotonic()-start>600:raise TimeoutError(tag)
    time.sleep(.25)
  finally:
   if p.poll() is None:p.kill();p.wait(timeout=20)
 text=errpath.read_text();raw=outpath.read_text().strip().strip(',');ids=[int(v) for v in raw.split(',') if v.strip()] if p.returncode==0 else []
 if tokens==128 and cachegb==32:LONG_REFERENCE[prompt]=ids
 expected=REFERENCE[prompt] if tokens==64 else LONG_REFERENCE[prompt]
 m=re.search(r'decode: (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\)',text);r1=resource.getrusage(resource.RUSAGE_CHILDREN)
 row={'tag':tag,'prompt':prompt,'cache_GB':cachegb,'exit':p.returncode,'tokens':ids,'exact_agreement':ids==expected,'requested_tokens':tokens,'metrics':[l for l in text.splitlines() if any(k in l for k in ['THP advice','decode:','decode per token:','store:','token latency','dspark:'])],'snapshot':snapshot,'pressure_before':before,'pressure_after':pressure(),'cgroup_before':cgbefore,'cgroup_after':cgstats(),'wall_seconds':time.monotonic()-start,'child_user_seconds':r1.ru_utime-r0.ru_utime,'child_system_seconds':r1.ru_stime-r0.ru_stime,'child_major_faults':r1.ru_majflt-r0.ru_majflt,'child_minor_faults':r1.ru_minflt-r0.ru_minflt}
 if m:row.update(decode_ms=float(m[2]),reported_tps=float(m[3]))
 REPORT['runs'].append(row);write();print(tag,'exit',p.returncode,'tps',row.get('reported_tps'),'exact',row['exact_agreement'],flush=True)
 if p.returncode or not m or not row['exact_agreement'] or snapshot is None:raise RuntimeError('Correctness/execution gate failed: '+tag)

def main():
 guard();REPORT['binary_sha256']=hashlib.sha256(Path(EXE).read_bytes()).hexdigest()
 state={'restore_model':json.loads(request('/health')).get('active_model'),'phase':'unloading API'};(ROOT/'state.json').write_text(json.dumps(state)+'\n')
 request('/v1/unload',{});subprocess.run(['systemctl','--user','stop','shiba-model-api.service'],check=True);state['phase']='benchmarking';(ROOT/'state.json').write_text(json.dumps(state)+'\n');write()
 for tag,prompt,cachegb in [('cache32-code-a','code',32),('cache32-prose-a','prose',32),('cache38-code-a','code',38),('cache38-prose-a','prose',38),('cache38-code-b','code',38),('cache38-prose-b','prose',38),('cache32-code-b','code',32),('cache32-prose-b','prose',32)]:run(tag,prompt,cachegb)
 # Extend only a consistent, pressure-free improvement to128-token checks.
 eligible=True
 for prompt in ['code','prose']:
  a=[v['reported_tps'] for v in REPORT['runs'] if v['prompt']==prompt and v['cache_GB']==32]
  b=[v['reported_tps'] for v in REPORT['runs'] if v['prompt']==prompt and v['cache_GB']==38]
  eligible=eligible and min(b)>max(a)*1.03 and sum(b)/sum(a)>1.05
 for row in REPORT['runs']:
  fields=dict(l.split() for l in row['cgroup_after'].get('memory.events','').splitlines())
  eligible=eligible and fields.get('high')=='0' and fields.get('oom')=='0'
 REPORT['long_confirmation_eligible']=eligible;write()
 if eligible:
  for prompt in ['code','prose']:
   run('long32-'+prompt,prompt,32,128);run('long38-'+prompt,prompt,38,128)
 REPORT['passed']=True;write();state['phase']='complete; restoration pending';(ROOT/'state.json').write_text(json.dumps(state)+'\n')
if __name__=='__main__':main()
