import importlib.util,json,subprocess,threading,time
from pathlib import Path
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gate',root/'run-gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
state={'restore_model':json.loads(g.request('/health')).get('active_model')};(root/'state.json').write_text(json.dumps(state))
g.request('/v1/unload',{});subprocess.run(['systemctl','--user','stop','shiba-model-api.service'],check=True)
g.run('off','code',64,0,True)
def ready():
 p=root/'down-compact-code-64-1.tokens'
 while not (root/'results.json').exists() or len(g.REPORT['runs'])<2:
  if p.exists() and len([x for x in p.read_text().strip(',\n ').split(',') if x.strip()])>=8:
   (root/'KILL_WORKER_NOW').write_text('Eight output tokens committed; terminate only secondary worker PID155.\n');return
  time.sleep(.02)
threading.Thread(target=ready,daemon=True).start()
try:g.run('down-compact','code',64,1,True)
except RuntimeError as e:
 row=g.REPORT['runs'][-1]
 if not(row['worker_failed'] and row['exact_agreement'] and row['exit']==0):raise
row=g.REPORT['runs'][-1];assert row['worker_failed'] and row['exact_agreement'] and len(row['tokens'])==64
g.REPORT['passed']=True;g.REPORT['contract']='Forced worker loss after >=8 committed tokens; full 64-token exact agreement required; inspect lazy refill counters separately';g.save()
