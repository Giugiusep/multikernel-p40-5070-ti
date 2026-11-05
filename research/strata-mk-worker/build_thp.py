"""Build an isolated host-arena candidate using unchanged reference GPU objects."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent/'Strata-mk-worker/src/core/pinned.cu'
REFERENCE=ROOT.parent/'Strata/build/libstrata_core.a'
OUT=ROOT/'primary-thp-build'
OUT.mkdir(exist_ok=True)
obj=OUT/'pinned.cu.o'
cmd=['/usr/bin/g++-13','-x','c++','-O3','-DNDEBUG','-std=c++20',
     '-I'+str(ROOT.parent/'Strata-mk-worker/include'),
     '-isystem','/usr/local/cuda-12.8/targets/x86_64-linux/include','-c',str(SRC),'-o',str(obj)]
subprocess.run(cmd,check=True)
lib=OUT/'libstrata_core.a'
shutil.copy2(REFERENCE,lib)
subprocess.run(['ar','r',str(lib),str(obj)],check=True)
base=json.loads((ROOT/'primary-build/build-manifest.json').read_text())
link=base['link'][:]
link[link.index('-o')+1]=str(OUT/'strata')
assert link.count(str(REFERENCE))==1
link[link.index(str(REFERENCE))]=str(lib)
subprocess.run(link,check=True)
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'build-manifest.json').write_text(json.dumps({'compile':cmd,'link':link,'source_sha256':sha(SRC),'original_core_sha256':sha(REFERENCE),'candidate_core_sha256':sha(lib),'binary_sha256':sha(OUT/'strata'),'base':base},indent=2)+'\n')
print(OUT/'strata')
