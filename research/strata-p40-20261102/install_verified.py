import hashlib,json,pathlib,sys
root=pathlib.Path(__file__).resolve().parent
source=pathlib.Path('/home/shiba/multikernel-experiment/research/Strata-dsv41-experiment')
sys.path.insert(0,str(source))
from strata_control import updater,engine_component as ec,signing
archive=root/'strata-engine-0.1.55-linux-x86_64-cuda-cuda12.tar.gz'
expected='2455898afd445fa2c814676f0923339707344bc198286f14694b76c9a7baefb6'
with archive.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==expected
folder=root/'engine';updater.extract(archive,folder,'tar.gz',6*1024**3)
component=ec.read_manifest(folder)
signing.install_component_verifier(signing.load_trusted_keys())
ec.verify_signature(component,{ec.ENV_REQUIRE_SIGNED:'1'})
ec.verify(component,full=True)
assert component.entrypoint=='strata' and 'sm_61' in component.cuda_archs,component.cuda_archs
(root/'engine-verification.json').write_text(json.dumps({'archive_sha256':actual,'signature_and_files_verified':True,'archs':component.cuda_archs,'executable':str(component.exe)},indent=2)+'\n')
print(component.cuda_archs)
