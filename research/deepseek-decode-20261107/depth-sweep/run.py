"""Paired DSpark verification cap test; no arithmetic or expert-count changes."""
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run_benchmark as b
b.ROOT=HERE
b.REPORT={'contract':'deployed engine; greedy; six experts; confidence0.65; context8192; only verification cap changes','runs':[],'passed':False}
b.guard()
state={'restore_model':json.loads(b.request('/health')).get('active_model'),'phase':'unloading API'}
(HERE/'state.json').write_text(json.dumps(state)+'\n')
b.request('/v1/unload',{});subprocess.run(['systemctl','--user','stop','shiba-model-api.service'],check=True)
state['phase']='benchmarking';(HERE/'state.json').write_text(json.dumps(state)+'\n')
for cap,tag,prompt in [(5,'cap5-code-a','code'),(5,'cap5-prose-a','prose'),(3,'cap3-code-a','code'),(3,'cap3-prose-a','prose'),(3,'cap3-code-b','code'),(3,'cap3-prose-b','prose'),(5,'cap5-code-b','code'),(5,'cap5-prose-b','prose')]:
 b.CFG['env']['DS4_DSPARK_MAX']=str(cap)
 b.CFG['env']['DS4_CPU_LOG']='1'
 row=b.run(tag,b.CFG['exe'],prompt=prompt);row['verification_draft_cap']=cap;b.save()
b.REPORT['passed']=True;b.save();state['phase']='complete; restoration pending';(HERE/'state.json').write_text(json.dumps(state)+'\n')
