import concurrent.futures,hashlib,json,pathlib,subprocess,time
root=pathlib.Path('/home/shiba/models/deepseek-v41-shared');root.mkdir(exist_ok=True)
items=json.load(open('/home/shiba/multikernel-experiment/model-downloads/deepseek-companions-manifest.json'))
def run(x):
 target=root/x['name'];part=root/(x['name']+'.part')
 if not target.exists():
  print('Downloading '+x['name'],flush=True)
  subprocess.run(['curl','-fLsS','--retry','8','--retry-delay','5','--connect-timeout','30','-C','-','-o',str(part),f"https://huggingface.co/{x['repository']}/resolve/{x['revision']}/{x['path']}?download=true"],check=True)
  source=part
 else:source=target
 assert source.stat().st_size==x['bytes']
 print('Verifying '+x['name'],flush=True);h=hashlib.sha256()
 with source.open('rb') as f:
  while b:=f.read(16*1024*1024):h.update(b)
 assert h.hexdigest()==x['sha256'],'HASH MISMATCH '+x['name']
 if source==part:part.rename(target)
 print('VERIFIED '+x['name'],flush=True);return dict(x,verified=True)
with concurrent.futures.ThreadPoolExecutor(2) as pool:results=list(pool.map(run,items))
(root/'provenance.json').write_text(json.dumps({'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':results},indent=2)+'\n');print('ALL COMPANIONS VERIFIED',flush=True)
