#!/usr/bin/env python3
"""Register the existing model API using Studio's own encrypted provider storage."""
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

BACKEND=Path('/home/shiba/.unsloth/studio/unsloth_studio/lib/python3.13/site-packages/studio/backend')
sys.path.insert(0,str(BACKEND))
from storage import providers_db, credential_secrets, studio_db

ROOT=Path('/home/shiba/multikernel-experiment/model-api')
key=(ROOT/'api-key').read_text().strip()
req=urllib.request.Request('http://127.0.0.1:19080/v1/models',headers={'Authorization':'Bearer '+key})
with urllib.request.urlopen(req,timeout=30) as response:
    catalog=json.load(response)['data']
models=[m['id'] for m in catalog if m.get('metadata',{}).get('available')]
if not models:
    raise SystemExit('no available models; refusing to replace saved selection')
# SQLite backup includes any live WAL content. Contains private data: never commit.
backup=Path('/home/shiba/.unsloth/studio/studio-before-shiba-'+time.strftime('%Y%m%d-%H%M%S')+'.db')
with sqlite3.connect('/home/shiba/.unsloth/studio/studio.db') as source, sqlite3.connect(backup) as target:
    source.backup(target)
backup.chmod(0o600)
provider='shiba-local-models'
if not providers_db.get_provider(provider):
    providers_db.create_provider(id=provider,provider_type='custom',display_name='Shiba local models (Strata / Multikernel / RTX)',base_url='http://127.0.0.1:19080/v1',models=models,available_models=models)
with providers_db.provider_bundle_transaction() as connection:
    providers_db.update_provider(provider,models=models,available_models=models,is_enabled=True,connection=connection)
    credential_secrets.save_provider_api_key(provider,key,connection=connection)
folder=studio_db.add_scan_folder('/home/shiba/models')
assert credential_secrets.get_provider_api_key(provider)==key
assert providers_db.get_provider(provider)['models']==models
report={'provider_id':provider,'base_url':'http://127.0.0.1:19080/v1','models':models,'count':len(models),'scan_folder':folder,'credential_verified':True}
Path('/home/shiba/multikernel-experiment/studio-integration/registration.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
