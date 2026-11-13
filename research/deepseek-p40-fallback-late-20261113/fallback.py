import importlib.util,json,subprocess,threading,time,socket,struct
from pathlib import Path
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gate',root/'run-gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
state={'restore_model':json.loads(g.request('/health')).get('active_model')};(root/'state.json').write_text(json.dumps(state))
g.request('/v1/unload',{});subprocess.run(['systemctl','--user','stop','shiba-model-api.service'],check=True)
g.REPORT['runs']=[json.loads((root.parent/'deepseek-p40-fallback-auto-20261113/results.json').read_text())['runs'][0]];g.save()
def ready():
 p=root/'down-compact-code-64-1.tokens'
 while not (root/'results.json').exists() or len(g.REPORT['runs'])<2:
  if p.exists() and len([x for x in p.read_text().strip(',\n ').split(',') if x.strip()])>=48:
   with socket.socket(socket.AF_VSOCK,socket.SOCK_STREAM) as ctl:
    ctl.setsockopt(socket.AF_VSOCK,9,1);ctl.settimeout(5);ctl.connect((1,6005));ctl.sendall(struct.pack('<I',0x4b494c4c));assert ctl.recv(4)==b'\0'*4
   (root/'KILL_WORKER_NOW').write_text('Automatic termination acknowledged after 48 committed tokens.\n');return
  time.sleep(.02)
threading.Thread(target=ready,daemon=True).start()
try:g.run('down-compact','code',64,1,True)
except RuntimeError as e:
 row=g.REPORT['runs'][-1]
 if not(row['worker_failed'] and row['exact_agreement'] and row['exit']==0):raise
row=g.REPORT['runs'][-1];assert row['worker_failed'] and row['exact_agreement'] and len(row['tokens'])==64
g.REPORT['passed']=True;g.REPORT['contract']='Forced worker loss after >=48 committed tokens; full 64-token exact agreement required; inspect lazy refill counters separately';g.save()
