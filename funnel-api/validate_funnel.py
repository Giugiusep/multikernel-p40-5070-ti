"""Check Funnel authentication without printing bearer credentials."""
import json,secrets,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path('/home/shiba/multikernel-experiment/model-api')
PUBLIC='https://shiba2th-1.tail0e03fa.ts.net:8443'

def check(base,key,expected):
    headers={'Authorization':'Bearer '+key} if key else {}
    req=urllib.request.Request(base+'/v1/models',headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=30) as r:
            status=r.status;data=json.load(r)
    except urllib.error.HTTPError as e:
        status=e.code;data=json.load(e)
    assert status==expected,(base,status,expected)
    return len(data.get('data',[]))

for attempt in range(20):
    try:
        with urllib.request.urlopen('http://127.0.0.1:19080/health',timeout=5):pass
        break
    except urllib.error.URLError:time.sleep(1)
results={}
for name,base in [('local','http://127.0.0.1:19080'),('funnel',PUBLIC)]:
    check(base,None,401);check(base,'incorrect-secret',401)
    counts={}
    for label,file in [('owner',ROOT/'api-key'),('user-2',ROOT/'api-keys/user-2.key')]:
        counts[label]=check(base,file.read_text().strip(),200)
    results[name]={'missing_key':401,'wrong_key':401,'model_counts':counts}
# Test immediate revocation with a disposable third credential, never the real users' keys.
f=ROOT/'api-keys/revocation-test.key';temporary=secrets.token_urlsafe(36)
try:
    f.touch(mode=0o600,exist_ok=False);f.write_text(temporary+'\n')
    check('http://127.0.0.1:19080',temporary,200)
finally:f.unlink(missing_ok=True)
check('http://127.0.0.1:19080',temporary,401)
results['immediate_revocation']='passed'
Path('/home/shiba/multikernel-experiment/funnel-api/validation.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
