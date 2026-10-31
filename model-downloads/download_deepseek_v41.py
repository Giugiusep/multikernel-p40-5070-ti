#!/usr/bin/env python3
"""Pinned, resumable direct downloads; no duplicated Hugging Face cache."""
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT=Path('/home/shiba/models/deepseek-v4.1-flash-1017-iq2-xs')
meta=json.loads(Path('/home/shiba/multikernel-experiment/model-downloads/deepseek-v41-metadata.json').read_text())
files=[f for f in meta['siblings'] if 'iq2_xs' in f['rfilename'].lower()]
assert len(files)==4
remaining=sum(max(0,f['size']-(ROOT/(Path(f['rfilename']).name+'.part')).stat().st_size) if (ROOT/(Path(f['rfilename']).name+'.part')).exists() else (0 if (ROOT/Path(f['rfilename']).name).exists() else f['size']) for f in files)
assert shutil.disk_usage(ROOT).free>remaining+10*2**30,'Insufficient space with 10GiB reserve'

def download(f):
    name=Path(f['rfilename']).name
    target=ROOT/name;part=ROOT/(name+'.part')
    if not target.exists():
        url=f"https://huggingface.co/{meta['id']}/resolve/{meta['sha']}/{f['rfilename']}?download=true"
        print('Downloading '+name,flush=True)
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','8','--retry-delay','5','--connect-timeout','30','--continue-at','-','--output',str(part),url],check=True)
        assert part.stat().st_size==f['size'],name+' size mismatch'
        source=part
    else:source=target
    print('Verifying '+name,flush=True)
    assert source.stat().st_size==f['size']
    h=hashlib.sha256()
    with source.open('rb') as stream:
        while block:=stream.read(16*1024*1024):h.update(block)
    assert h.hexdigest()==f['lfs']['sha256'],name+' HASH MISMATCH'
    if source==part:part.rename(target)
    print('VERIFIED '+name,flush=True)
    return {'file':name,'bytes':f['size'],'sha256':h.hexdigest(),'verified':True}

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    verified=list(executor.map(download,files))
report={'repository':meta['id'],'revision':meta['sha'],'total_bytes':sum(f['size'] for f in files),'files':verified,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(ROOT/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print('COMPLETE: all four shards hash verified',flush=True)
