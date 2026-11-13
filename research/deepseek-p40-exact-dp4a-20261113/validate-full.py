import importlib.util,json,struct,socket,subprocess,os,ctypes
from pathlib import Path
import numpy as np
import statistics,time
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('oracle',ROOT.parent/'deepseek-p40-integration-20261108/validate-q8k.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
class Client(v.Client):
 def __init__(self):
  self.s=socket.socket(socket.AF_VSOCK,socket.SOCK_STREAM);self.s.setsockopt(socket.AF_VSOCK,9,1);self.s.settimeout(15);self.s.connect((1,int(os.environ.get('TEST_PORT','6014'))));self.seq=0;self.short=0
libm=ctypes.CDLL('libm.so.6');libm.expf.argtypes=[ctypes.c_float];libm.expf.restype=ctypes.c_float
files,cases=v.layout();chosen={}
for l,r in cases:
 if r[1]['type'] in ['IQ2_XXS','IQ2_XS','IQ2_S'] and r[2]['type'] in ['IQ2_XXS','IQ2_XS','IQ2_S']:chosen.setdefault(r[1]['type'],(l,r))
assert len(chosen)==3
fixture=ROOT/'fixtures';fixture.mkdir(exist_ok=True);report={'passed':False,'cases':[]};c=Client();rng=np.random.default_rng(20261113);env=os.environ.copy();env['OMP_NUM_THREADS']='1'
def ref(ty,k,n,m,w,x,name):
 (fixture/'w.bin').write_bytes(w);x.astype('<f4').tofile(fixture/'x.f32');out=fixture/name
 subprocess.run([str(ROOT/'matvec-reference'),str(ty),str(k),str(n),str(m),str(fixture/'w.bin'),str(fixture/'x.f32'),str(out)],check=True,env=env)
 return out.read_bytes(),out.with_name(name+'.q8k').read_bytes()
def packed(raw,m,n):
 q=bytearray();sc=[]
 for i in range(m*n//256):
  d=struct.unpack_from('<f',raw,i*292)[0];q.extend(raw[i*292+4:i*292+260]);sc.extend([d]*8)
 return q+np.asarray(sc,dtype='<f4').tobytes()
for ty,(l,roles) in chosen.items():
 w={}
 for role in [1,3,2]:
  r=roles[role]
  with files[r['file']].open('rb') as f:f.seek(r['offset']);w[role]=f.read(r['bytes'])
 tg=v.TYPES[ty];td=v.TYPES[roles[2]['type']];blob=w[1]+w[3]+w[2];c.request(1,struct.pack('<8I',l,0,tg,td,5120,2304,len(w[1]),len(w[2]))+blob)
 for m in range(1,9):
  x=rng.standard_normal((m,5120)).astype('<f4');x[0,:256]=0
  gr,raw=ref(tg,5120,2304,m,w[1],x,'gate');ur,_=ref(tg,5120,2304,m,w[3],x,'up')
  out=c.request(6,struct.pack('<4I',l,0,m,0)+packed(raw,m,5120));gate_exact=bytes(out[28:28+m*2304*4])==gr;up_exact=bytes(out[28+m*2304*4:])==ur
  row={'gate_type':ty,'down_type':roles[2]['type'],'M':m,'gate_exact':gate_exact,'up_exact':up_exact};report['cases'].append(row);(ROOT/('validation-'+os.environ.get('TEST_PORT','6014')+'.json')).write_text(json.dumps(report,indent=2));print(row,flush=True);assert gate_exact and up_exact,row
  g=np.minimum(np.frombuffer(gr,dtype='<f4'),np.float32(10));u=np.clip(np.frombuffer(ur,dtype='<f4'),np.float32(-10),np.float32(10));h=np.empty_like(g)
  for i in range(len(h)):h[i]=(g[i]/np.float32(np.float32(1)+np.float32(libm.expf(float(-g[i])))))*u[i]
  yr,raw=ref(td,2304,5120,m,w[2],h.reshape(m,2304),'down');out=c.request(4,struct.pack('<4I',l,0,m,0)+packed(raw,m,2304));row['down_exact']=bytes(out[28:])==yr;(ROOT/('validation-'+os.environ.get('TEST_PORT','6014')+'.json')).write_text(json.dumps(report,indent=2));assert row['down_exact'],row
  gut=[];dt=[];rt=[]
  for repeat in range(23):
   t=time.perf_counter_ns();o=c.request(6,struct.pack('<4I',l,0,m,0)+packed((fixture/'gate.q8k').read_bytes(),m,5120));assert bytes(o[28:28+m*2304*4])==gr and bytes(o[28+m*2304*4:])==ur
   a=sum(struct.unpack_from('<3f',o,16));o=c.request(4,struct.pack('<4I',l,0,m,0)+packed(raw,m,2304));assert bytes(o[28:])==yr
   if repeat>=3:gut.append(a);dt.append(sum(struct.unpack_from('<3f',o,16)));rt.append((time.perf_counter_ns()-t)/1e6)
  row.update(gu_gpu_us=statistics.median(gut),down_gpu_us=statistics.median(dt),combined_rpc_ms=statistics.median(rt),combined_rpc_p95_ms=sorted(rt)[18]);(ROOT/('validation-'+os.environ.get('TEST_PORT','6014')+'.json')).write_text(json.dumps(report,indent=2))

 c.request(3)
report['passed']=True;(ROOT/('validation-'+os.environ.get('TEST_PORT','6014')+'.json')).write_text(json.dumps(report,indent=2)+'\n');c.s.close()
for p in fixture.iterdir():p.unlink()
