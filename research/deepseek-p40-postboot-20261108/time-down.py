import importlib.util,json,struct,subprocess,os
from pathlib import Path
import numpy as np
import time
import statistics
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('worker_oracle',ROOT.parent/'deepseek-p40-integration-20261108/validate-q8k.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
files,cases=v.layout();chosen={}
for l,r in cases:
 if r[2]['type'] in ['IQ2_XXS','IQ2_XS','IQ2_S']:chosen.setdefault(r[2]['type'],(l,r))
fixtures=ROOT/'down-fixtures';fixtures.mkdir(exist_ok=True)
result={'passed':False,'cases':[]};c=v.Client();rng=np.random.default_rng(20261108)
for ty,(layer,roles) in chosen.items():
 blob=b'';dn=None
 for role in [1,3,2]:
  r=roles[role]
  with files[r['file']].open('rb') as f:f.seek(r['offset']);b=f.read(r['bytes'])
  assert len(b)==r['bytes'];blob+=b
  if role==2:dn=b
 (fixtures/'down.bin').write_bytes(dn)
 c.request(1,struct.pack('<8I',layer,0,v.TYPES[roles[1]['type']],v.TYPES[ty],5120,2304,roles[1]['bytes'],roles[2]['bytes'])+blob)
 for m in range(1,9):
  x=rng.standard_normal((m,2304)).astype('<f4');x[0,:256]=0;x.tofile(fixtures/'x.f32')
  env=os.environ.copy();env['OMP_NUM_THREADS']='1'
  subprocess.run([str(ROOT/'down-reference'),str(v.TYPES[ty]),'5120',str(m),str(fixtures/'down.bin'),str(fixtures/'x.f32'),str(fixtures/'ref.f32')],check=True,env=env)
  raw=(fixtures/'ref.f32.q8k').read_bytes();q=bytearray();sc=[]
  for i in range(m*9):
   d=struct.unpack_from('<f',raw,i*292)[0];q.extend(raw[i*292+4:i*292+260]);sc.extend([d]*8)
  payload=struct.pack('<4I',layer,0,m,0)+q+np.asarray(sc,dtype='<f4').tobytes()
  expected=(fixtures/'ref.f32').read_bytes();start=time.perf_counter_ns();b=c.request(4,payload);elapsed=(time.perf_counter_ns()-start)/1e6;ts,y,*_=v.parse(b,m,0)
  samples=[];gpu=[]
  for rep in range(23):
   start=time.perf_counter_ns();rb=c.request(4,payload);t=(time.perf_counter_ns()-start)/1e6;st,yy,*_=v.parse(rb,m,0);assert yy.tobytes()==expected
   if rep>=3:samples.append(t);gpu.append(sum(st))
  actual=y.tobytes();ref=np.frombuffer(expected,dtype='<f4');row={'first_roundtrip_ms':elapsed,'median_roundtrip_ms':statistics.median(samples),'p95_roundtrip_ms':float(np.percentile(samples,95)),'median_GPU_us':statistics.median(gpu),'verified_repeats':24,'GPU_stage_us':list(ts),'type':ty,'layer':layer,'M':m,'bit_exact':actual==expected,'mismatched_values':int(np.count_nonzero(y.ravel().view('u4')!=ref.view('u4'))),'max_abs':float(np.max(np.abs(y.ravel()-ref)))};result['cases'].append(row);(ROOT/'down-timing-results.json').write_text(json.dumps(result,indent=2)+'\n');print(row,flush=True);assert actual==expected,row
 c.request(3)
result['passed']=True;(ROOT/'down-timing-results.json').write_text(json.dumps(result,indent=2)+'\n');c.s.close()
for p in fixtures.iterdir():p.unlink()
