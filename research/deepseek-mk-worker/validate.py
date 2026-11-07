"""Independent real-weight oracle and bounded DeepSeek Multikernel client."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import statistics
import struct
import subprocess
import time
import numpy as np
ROOT=Path(__file__).resolve().parent
PACK=Path('/home/shiba/models/deepseek-v4.1-flash-1017-iq2-xs/strata-pack')
D,F=5120,2304
MAGIC=0x4458454d
FRAME=struct.Struct('<6I')
TYPES={'IQ2_XXS':16,'IQ2_XS':17,'IQ2_S':22,'IQ3_XXS':18}
class Client:
 def __init__(self):
  self.s=socket.socket(socket.AF_VSOCK,socket.SOCK_STREAM);self.s.setsockopt(socket.AF_VSOCK,9,1);self.s.settimeout(15);self.s.connect((1,6010));self.seq=0;self.short=0
 def send(self,data):
  v=memoryview(data)
  while v:
   try:n=self.s.send(v[:32768])
   except InterruptedError:continue
   if n<=0:raise RuntimeError('closed send')
   self.short+=n<min(len(v),32768);v=v[n:]
   if v:time.sleep(.0001)
 def recv(self,n):
  b=bytearray(n);v=memoryview(b)
  while v:
   try:k=self.s.recv_into(v)
   except InterruptedError:continue
   if not k:raise RuntimeError('closed receive')
   v=v[k:]
  return b
 def request(self,op,data=b'',error=None):
  self.seq+=1;self.send(FRAME.pack(MAGIC,1,op,0,self.seq,len(data)));self.send(data)
  m,v,o,status,seq,n=FRAME.unpack(self.recv(24));assert (m,v,o,seq)==(MAGIC,1,op,self.seq) and n<=16*1024*1024
  b=self.recv(n)
  if error is not None:
   assert status==1 and error in b.decode(),(status,b[:200]);return
  assert status==0,b[:200];return b

def quant(x):
 v=x.astype(np.float32).reshape(len(x),-1,32);a=np.max(np.abs(v),axis=2);sc=a/np.float32(127);safe=np.where(sc>0,sc,np.float32(1));q=np.rint(v/safe[:,:,None]).astype(np.int8);return q.reshape(len(x),-1),sc

def dot(w,x):
 # float64 independent accumulation over ggml-decoded weights, bounded conversion.
 y=np.empty((len(x),len(w)),np.float64);xh=x.astype(np.float64)
 for i in range(0,len(w),64):y[:,i:i+64]=xh@np.asarray(w[i:i+64],np.float64).T
 return y

def layout():
 files={};layers={}
 for line in (PACK/'tensors.txt').open():
  v=line.split()
  if v and v[0]=='file':
   path=(PACK/v[2]).resolve()
   if not path.exists():path=(PACK.parent/v[2]).resolve()
   files[int(v[1])]=path
  if len(v)<8:continue
  m=re.fullmatch(r'layers\.(\d+)\.ffn\.experts\.0\.w([123])\.weight',v[1])
  if m:layers.setdefault(int(m[1]),{})[int(m[2])]={'type':v[2],'file':int(v[3]),'offset':int(v[4]),'bytes':int(v[5]),'shape':list(map(int,v[6:]))}
 selected={}
 for l,r in layers.items():
  assert r[1]['type']==r[3]['type'] and r[1]['shape']==[F,D] and r[2]['shape']==[D,F]
  selected.setdefault((r[1]['type'],r[2]['type']),(l,r))
 return files,list(selected.values())

def decode_case(layer,roles,files):
 cache=ROOT/'fixtures';cache.mkdir(exist_ok=True);blob=b'';weights={};meta=[]
 for role in (1,3,2):
  r=roles[role];raw=cache/f'l{layer}-w{role}.bin'
  with files[r['file']].open('rb') as h:h.seek(r['offset']);b=h.read(r['bytes'])
  assert len(b)==r['bytes'];raw.write_bytes(b);blob+=b
  dec=raw.with_suffix('.f32');rows,k=r['shape'];subprocess.run([str(ROOT/'decode'),str(TYPES[r['type']]),str(rows),str(k),str(raw),str(dec)],check=True)
  weights[role]=np.memmap(dec,dtype='<f4',mode='r',shape=(rows,k));assert np.isfinite(weights[role]).all()
  meta.append({'role':role,**r,'sha256':hashlib.sha256(b).hexdigest()})
 return blob,weights,meta

def parse(b,rows,debug):
 assert struct.unpack_from('<4I',b)==(rows,debug,D,F)
 ts=struct.unpack_from('<3f',b,16);off=28
 def take(dtype,n):
  nonlocal off
  v=np.frombuffer(b,dtype=dtype,count=n,offset=off).copy();off+=v.nbytes;return v
 y=take('<f4',rows*D).reshape(rows,D);h=hq=hd=None
 if debug:h=take('<f4',rows*F).reshape(rows,F);hq=take('i1',rows*F).reshape(rows,F);hd=take('<f4',rows*F//32).reshape(rows,F//32)
 assert off==len(b) and np.isfinite(y).all() and all(np.isfinite(ts))
 return ts,y,h,hq,hd

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--iterations',type=int,default=20);ap.add_argument('--max-cases',type=int,default=20);args=ap.parse_args();assert 3<=args.iterations<=100
 files,cases=layout();c=Client();info=json.loads(c.request(0));assert info['contract']=='deepseek-iq-q8-32-v1';results={'worker':info,'cases':[],'passed':False}
 (ROOT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 c.request(99,error='unknown');c.request(2,struct.pack('<4I',0,0,1,0)+bytes(D+D//32*4),error='not resident')
 rng=np.random.default_rng(20261107)
 for layer,roles in cases[:args.max_cases]:
  blob,w,meta=decode_case(layer,roles,files);gu=roles[1]['bytes'];dn=roles[2]['bytes'];load=struct.pack('<8I',layer,0,TYPES[roles[1]['type']],TYPES[roles[2]['type']],D,F,gu,dn)+blob
  c.request(1,load[:-1],error='byte count');c.request(1,struct.pack('<I',40)+load[4:],error='layout');c.request(1,load);c.request(1,load,error='duplicate')
  case={'layer':layer,'roles':meta,'widths':[]};results['cases'].append(case)
  for rows in range(1,9):
   x=rng.standard_normal((rows,D)).astype(np.float32);x[0,:32]=0;q,sc=quant(x);xh=(q.reshape(rows,-1,32).astype(np.float64)*sc[:,:,None]).reshape(rows,D)
   request=struct.pack('<4I',layer,0,rows,1)+q.tobytes()+sc.tobytes();b=c.request(2,request);times,y,h,hq,hd=parse(b,rows,1)
   g=dot(w[1],xh);u=dot(w[3],xh);g=np.minimum(g,10);u=np.clip(u,-10,10);refh=g/(1+np.exp(-g))*u
   hrel=float(np.abs(h-refh).sum()/(np.abs(refh).sum()+1e-30));hmax=float(np.max(np.abs(h-refh)/(1+np.abs(refh))));assert hrel<5e-5 and hmax<2e-4,(layer,rows,'GU',hrel,hmax)
   eq,es=quant(h);assert np.array_equal(hd,es),(layer,rows,'intermediate scales');assert np.array_equal(hq,eq),(layer,rows,'intermediate codes')
   hv=(hq.reshape(rows,-1,32).astype(np.float64)*hd[:,:,None]).reshape(rows,F);refy=dot(w[2],hv)
   rel=float(np.abs(y-refy).sum()/(np.abs(refy).sum()+1e-30));mx=float(np.max(np.abs(y-refy)/(1+np.abs(refy))));assert rel<2e-5 and mx<2e-4,(layer,rows,'down',rel,mx)
   # Check final output against CPU-computed GU and intermediate quantization too.
   rq,rs=quant(refh.astype(np.float32));rh=(rq.reshape(rows,-1,32).astype(np.float64)*rs[:,:,None]).reshape(rows,F);end=dot(w[2],rh);endrel=float(np.abs(y-end).sum()/(np.abs(end).sum()+1e-30));assert endrel<1e-3,(layer,rows,'end-to-end',endrel)
   if rows>1:
    for row in range(rows):
     one=struct.pack('<4I',layer,0,1,1)+q[row:row+1].tobytes()+sc[row:row+1].tobytes()
     _,sy,sh,sq,ss=parse(c.request(2,one),1,1)
     for actual,expected in ((sy[0],y[row]),(sh[0],h[row]),(sq[0],hq[row]),(ss[0],hd[row])):
      assert actual.tobytes()==expected.tobytes(),(layer,rows,row,'batched/single-row mismatch')
   req=struct.pack('<4I',layer,0,rows,0)+q.tobytes()+sc.tobytes();samples=[];gpu=[]
   for i in range(args.iterations+3):
    start=time.perf_counter_ns();rb=c.request(2,req);elapsed=(time.perf_counter_ns()-start)/1e6;ts,yy,_,_,_=parse(rb,rows,0);assert yy.tobytes()==y.tobytes(),(layer,rows,'repeat')
    if i>=3:samples.append(elapsed);gpu.append(sum(ts))
   record={'M':rows,'GU_relative_L1':hrel,'GU_max_normalized':hmax,'down_relative_L1':rel,'down_max_normalized':mx,'end_to_end_relative_L1':endrel,'median_roundtrip_ms':statistics.median(samples),'p95_roundtrip_ms':float(np.percentile(samples,95)),'median_GPU_us':statistics.median(gpu),'gpu_stage_us_first':times,'verified_repeats':args.iterations+3,'single_row_bit_agreement':True}
   case['widths'].append(record);(ROOT/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(layer,rows,record,flush=True)
  c.request(2,struct.pack('<4I',layer,0,0,0),error='dimensions');badsc=np.full((1,D//32),np.nan,np.float32);c.request(2,struct.pack('<4I',layer,0,1,0)+bytes(D)+badsc.tobytes(),error='scale');c.request(3);c.request(2,struct.pack('<4I',layer,0,1,0)+bytes(D+D//32*4),error='not resident')
  del w
  for f in (ROOT/'fixtures').iterdir():f.unlink()
 results.update(passed=True,short_writes=c.short);(ROOT/'results.json').write_text(json.dumps(results,indent=2)+'\n');c.s.close()
if __name__=='__main__':main()
