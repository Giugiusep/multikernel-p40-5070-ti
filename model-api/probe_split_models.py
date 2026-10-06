"""Isolated post-upgrade local/split comparison; never changes API catalog."""
import argparse,glob,hashlib,json,os,signal,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXE='/home/shiba/llama.cpp/build-main-cuda-rpc-128/bin/llama-server'
PORT=19082
PROMPT='Write the integers from 11 through 100 in order, separated by spaces. Do not add an introduction or explanation.'
def request(path,payload=None):
 data=None if payload is None else json.dumps(payload).encode()
 req=urllib.request.Request(f'http://127.0.0.1:{PORT}'+path,data=data,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('--split',default='45,55');args=parser.parse_args()
 entry=next(e for e in json.loads((ROOT/'models.json').read_text()) if e['id']==args.model)
 model=entry.get('path') or glob.glob(entry['pattern'])[0]
 output=ROOT/(args.model+'-post-upgrade-split.json');results={'model':args.model,'context':4096,'split':args.split,'max_tokens':128,'prompt':PROMPT,'runs':[]}
 for mode in ['local','split']:
  log=ROOT/'logs'/(args.model+'-post-upgrade-'+mode+'.log')
  cmd=[EXE,'-m',model,'--host','127.0.0.1','--port',str(PORT),'-c','4096','-b','128','-ub','64','-ngl','16' if mode=='local' else '99','--fit','off','--no-ui','-np','1']
  if mode=='split':cmd+=['--rpc','mkvsock:1:5002','--device','CUDA0,RPC0','--split-mode','layer','--tensor-split',args.split]
  env=os.environ.copy();env['GGML_MK_VSOCK_NO_READ_SLEEP']='1';env['GGML_RPC_NO_RDMA']='1';env.pop('GGML_MK_VSOCK_NO_WRITE_SLEEP',None)
  record={'mode':mode,'command':cmd,'responses':[]};results['runs'].append(record);start=time.monotonic()
  with log.open('w') as f:
   proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    while True:
     if proc.poll() is not None:raise RuntimeError(f'server exited {proc.returncode}; see {log}')
     if time.monotonic()-start>1500:raise TimeoutError('load timed out')
     try:
      request('/health');break
     except OSError:time.sleep(2)
    record['load_seconds']=time.monotonic()-start
    print(json.dumps({'mode':mode,'loaded_seconds':record['load_seconds']}),flush=True)
    for index in range(3):
     r=request('/completion',{'prompt':PROMPT,'temperature':0,'seed':42,'n_predict':128,'cache_prompt':False})
     text=r['content'];row={'index':index,'text':text,'sha256':hashlib.sha256(text.encode()).hexdigest(),'timings':r.get('timings'),'tokens_predicted':r.get('tokens_predicted')};record['responses'].append(row)
     output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'mode':mode,**row}),flush=True)
   except Exception as exc:
    record['error']=str(exc);output.write_text(json.dumps(results,indent=2)+'\n');raise
   finally:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  output.write_text(json.dumps(results,indent=2)+'\n')
 local,remote=results['runs'];results['exact_measured_agreement']=all(a['text']==b['text'] for a,b in zip(local['responses'][1:],remote['responses'][1:]));results['stable_measured_outputs']=all(x['responses'][1]['text']==x['responses'][2]['text'] for x in results['runs']);output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'exact_measured_agreement':results['exact_measured_agreement'],'stable_measured_outputs':results['stable_measured_outputs']}),flush=True)
if __name__=='__main__':main()
