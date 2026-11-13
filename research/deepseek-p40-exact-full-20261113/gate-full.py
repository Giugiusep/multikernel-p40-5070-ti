import importlib.util,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gate',root/'run-gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
(root/'state.json').write_text(json.dumps({'restore_model':json.loads(g.request('/health')).get('active_model')}));g.request('/v1/unload',{});subprocess.run(['systemctl','--user','stop','shiba-model-api.service'],check=True)
g.REPORT['contract']='Full exact GPU gate/up + primary CPU SiLU/Q8_K + exact GPU down,128MiB admission, two32-token prompts, reference/off then byte-exact shadow then replacement; default/math/route/pacing unchanged'
os.environ['DS4_MK_PORT']='6013';os.environ['DS4_MK_CACHE_MIB']='128'
for prompt in ['code','prose']:
 for i,mode in enumerate(['off','full-exact-shadow','full-exact']):
  g.run(mode,prompt,32,i,True)
  if mode=='full-exact-shadow':
   row=g.REPORT['runs'][-1];assert row['shadow_count']>0 and row['max_shadow_rel_l1']==0,row
  if mode=='full-exact':assert g.REPORT['runs'][-1]['loads']>0
 g.save()
g.REPORT['passed']=True;g.save()
