import hashlib,json,pathlib,sys
source=pathlib.Path('/home/shiba/multikernel-experiment/research/Strata-dsv41-experiment');sys.path.insert(0,str(source))
from strata_control.dsv41_engine import install_package
archive=pathlib.Path('/home/shiba/multikernel-experiment/research/deepseek-v41-plan/strata-dsv41-0.1.55-linux-x86_64-cuda-cuda13.tar.gz')
lines=pathlib.Path('/home/shiba/multikernel-experiment/deepseek-v41/SHA256SUMS-v0.2.9.txt').read_text().splitlines()
expected=next(line.split()[0] for line in lines if line.split()[-1].lstrip('*')==archive.name)
with archive.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
assert actual==expected,'engine archive SHA mismatch'
exe=install_package(source/'engine-dsv41',archive,emit=print)
assert exe,'engine signature/package checks failed'
pathlib.Path('/home/shiba/multikernel-experiment/deepseek-v41/engine-install.json').write_text(json.dumps({'release':'v0.2.9','archive_sha256':actual,'executable':str(exe),'verified_signature_and_files':True},indent=2)+'\n')
