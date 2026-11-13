import importlib.util,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gate',root/'run-gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
(root/'state.json').write_text(json.dumps({'restore_model':json.loads(g.request('/health')).get('active_model')}))
g.request('/v1/unload',{});subprocess.run(['systemctl','--user','stop','shiba-model-api.service'],check=True)
g.REPORT['runs']=[json.loads((root.parent/'deepseek-p40-fallback-auto-20261113/results.json').read_text())['runs'][0]]
g.REPORT['contract']='512/1024/1024/512MiB P40 down-only cache; unchanged CPU32GB, six experts, DSpark, pacing; exact 64-token reference reused from same binary/config'
for i,mib in enumerate([512,1024,1024,512]):
 os.environ['DS4_MK_CACHE_MIB']=str(mib);g.run('down-compact','code',64,i,True);g.REPORT['runs'][-1]['cache_mib']=mib;g.save()
g.REPORT['passed']=True;g.save()
