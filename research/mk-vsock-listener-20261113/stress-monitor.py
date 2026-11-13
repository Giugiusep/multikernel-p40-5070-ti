import socket,struct,time,json,concurrent.futures,statistics
from pathlib import Path
out=Path(__file__).with_name('stress-results.json')
def conn(port):
 s=socket.socket(socket.AF_VSOCK,socket.SOCK_STREAM);s.setsockopt(socket.AF_VSOCK,9,1);s.settimeout(15);s.connect((1,port));return s
def read(s,n):
 b=bytearray()
 while len(b)<n:
  x=s.recv(min(32768,n-len(b)))
  if not x:raise RuntimeError('EOF')
  b.extend(x)
 return bytes(b)
def write(s,b):
 p=0
 while p<len(b):
  n=s.send(b[p:p+32768]);assert n>0;p+=n
  if p<len(b):time.sleep(.0001)
def echo(n,ident):
 b=(bytes((i+ident)%251 for i in range(251))*((n+250)//251))[:n]
 with conn(6004) as s:
  t=time.perf_counter_ns();write(s,struct.pack('<I',n));write(s,b);assert struct.unpack('<I',read(s,4))[0]==n;assert read(s,n)==b
  return (time.perf_counter_ns()-t)/1e6
r={'echo':{},'stress':{},'download':{}}
for n in [1024,8192,65536,524288,1048576,16777216]:r['echo'][n]=echo(n,19)
def lane(j):
 for i in range(500):echo([1,7,1024,8192,32768,65536,131071][i%7],i+j)
 return 500
def churn():
 for _ in range(500):
  with conn(6004):pass
 return 500
t=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 fs=[pool.submit(lane,j) for j in range(4)]+[pool.submit(churn)]
 r['stress']={'connections':sum(f.result() for f in fs),'seconds':time.monotonic()-t,'errors':0}
for n in [524288,1048576,16777216]:
 times=[];stats=[]
 for _ in range(10):
  with conn(6001) as s:
   s.sendall(struct.pack('<III',n,2,1));t=time.perf_counter_ns();b=read(s,n);elapsed=(time.perf_counter_ns()-t)/1e6;assert b==b'Z'*n;stats.append(struct.unpack('<5Q',read(s,40)));times.append(elapsed)
 r['download'][n]={'median_ms':statistics.median(times),'times_ms':times,'stats':stats,'MiB_s':n/1048576/(statistics.median(times)/1000)}
out.write_text(json.dumps(r,indent=2));print(json.dumps(r))
