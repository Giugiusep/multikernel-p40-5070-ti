import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'model-api'))
from validate_tts_api import request
subprocess.run(['systemctl','--user','start','shiba-model-api.service'],check=True)
for _ in range(45):
 try:
  json.loads(request('/health'));break
 except OSError:time.sleep(1)
else:raise RuntimeError('API restoration health timeout')
p=ROOT/'state.json';d=json.loads(p.read_text()) if p.exists() else {}
if d.get('restore_model'):request('/v1/switch',{'model':d['restore_model']})
(ROOT/'api-restoration.json').write_text(json.dumps(json.loads(request('/health')),indent=2)+'\n')
