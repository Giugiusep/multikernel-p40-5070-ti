"""Allocation/integrity probe: repeated real weight fixture, NOT model residency."""
import importlib.util,json,socket,struct,time,subprocess,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('oracle',ROOT.parent/'deepseek-p40-integration-20261108/validate-q8k.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
class Client(v.Client):
 def __init__(self):
  self.s=socket.socket(socket.AF_VSOCK,socket.SOCK_STREAM);self.s.setsockopt(socket.AF_VSOCK,9,1);self.s.settimeout(15);self.s.connect((1,6012));self.seq=0;self.short=0
files,cases=v.layout();layer,roles=next((l,r) for l,r in cases if r[2]['type']=='IQ2_XS');r=roles[2];ty=v.TYPES[r['type']]
with files[r['file']].open('rb') as f:f.seek(r['offset']);dn=f.read(r['bytes'])
assert len(dn)==r['bytes'];fixture=ROOT/'capacity-fixtures';fixture.mkdir(exist_ok=True);(fixture/'down.bin').write_bytes(dn)
x=np.random.default_rng(20261113).standard_normal((1,2304)).astype('<f4');x.tofile(fixture/'x.f32');env=os.environ.copy();env['OMP_NUM_THREADS']='1'
subprocess.run([str(ROOT.parent/'deepseek-p40-postboot-20261108/down-reference'),str(ty),'5120','1',str(fixture/'down.bin'),str(fixture/'x.f32'),str(fixture/'ref.f32')],env=env,check=True)
expected=(fixture/'ref.f32').read_bytes();raw=(fixture/'ref.f32.q8k').read_bytes();q=bytearray();sc=[]
for i in range(9):
 d=struct.unpack_from('<f',raw,i*292)[0];q.extend(raw[i*292+4:i*292+260]);sc.extend([d]*8)
activation=q+np.asarray(sc,dtype='<f4').tobytes();limit=22528*1024*1024;count=limit//len(dn);assert count<40*384
report={'passed':False,'synthetic_capacity_only':True,'fixture_type':r['type'],'fixture_bytes':len(dn),'planned_allocations':count,'payload_bytes':count*len(dn),'checkpoints':[]}
c=Client();assert json.loads(c.request(0))['cache_bytes']==limit;t=time.monotonic()
for i in range(count):
 l,e=divmod(i,384);c.request(5,struct.pack('<8I',l,e,ty,ty,5120,2304,0,len(dn))+dn)
 if (i+1)%512==0 or i==count-1:
  row={'allocations':i+1,'resident_bytes':(i+1)*len(dn),'elapsed_s':time.monotonic()-t};report['checkpoints'].append(row);(ROOT/'capacity-progress.json').write_text(json.dumps(report,indent=2));print(row,flush=True)
for i in [0,count//2,count-1]:
 l,e=divmod(i,384);out=c.request(4,struct.pack('<4I',l,e,1,0)+activation);assert bytes(out[28:])==expected
report['correctness_checks']=3;report['passed']=True;report['elapsed_s']=time.monotonic()-t;(ROOT/'capacity-ALLOCATED').write_text('22GiB fixture allocated; awaiting VRAM observation\n');(ROOT/'capacity-results.json').write_text(json.dumps(report,indent=2));print('CAPACITY_VERIFIED_WAITING_FOR_OBSERVATION',flush=True)
# Hold briefly for independent NVML observation, then free all allocations.
for _ in range(300):
 if (ROOT/'capacity-RELEASE').exists():break
 time.sleep(.1)
c.request(3);c.s.close()
for p in fixture.iterdir():p.unlink()
print('CAPACITY_FREED',flush=True)
