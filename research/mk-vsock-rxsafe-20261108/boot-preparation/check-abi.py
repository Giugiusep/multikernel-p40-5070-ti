"""Require the candidate to accept the preserved primary and secondary modules."""
from pathlib import Path
import hashlib,json,re,subprocess,gzip
BASE=Path('/home/shiba/multikernel-experiment')
PREP=BASE/'research/mk-vsock-rxsafe-20261108/boot-preparation'
BUILD=BASE/'research/linux-rxsafe-build'
RELEASE='7.0.0-mk2-shiba-mk1'
def symvers(path):
 return {p[1]:p[0].lower() for line in path.read_text().splitlines() if len(p:=line.split())>=2}
def mod_data(path,flag):
 result=subprocess.run(['modprobe',flag,str(path)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if result.returncode and flag=='--show-exports' and 'No data available' in result.stderr:return {}
 if result.returncode:raise RuntimeError(result.stderr)
 text=result.stdout
 return {p[1]:p[0].lower() for line in text.splitlines() if len(p:=line.split())==2 and p[0].startswith('0x')}
def config(path):
 d={}
 for line in path.read_text().splitlines():
  if line.startswith('CONFIG_'):k,v=line.split('=',1);d[k]=v
  elif line.startswith('# CONFIG_') and line.endswith(' is not set'):d[line[2:-11]]='n'
 return d
before=json.loads((PREP/'before.json').read_text())
for name,meta in before['preserved_files'].items():
 with Path(name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 assert actual==meta['sha256'],f'Baseline changed: {name}'
assert (BUILD/'include/config/kernel.release').read_text().strip()==RELEASE
old=config(Path('/boot')/('config-'+RELEASE));new=config(BUILD/'.config')
changes={k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
allowed={'CONFIG_CC_VERSION_TEXT','CONFIG_GCC_VERSION','CONFIG_AS_VERSION','CONFIG_LD_VERSION'}
# The upgraded guest exposes different host-tool capabilities. Allow those only
# where the corresponding runtime feature is disabled or unchanged.
assert old.get('CONFIG_RUST','n')!='y' and new.get('CONFIG_RUST','n')!='y'
allowed.update(k for k in changes if k.startswith('CONFIG_RUSTC_'))
assert old.get('CONFIG_DEBUG_INFO_BTF','n')!='y' and new.get('CONFIG_DEBUG_INFO_BTF','n')!='y'
allowed.add('CONFIG_PAHOLE_VERSION')
assert old.get('CONFIG_MODULE_SIG_KEY_TYPE_RSA')==new.get('CONFIG_MODULE_SIG_KEY_TYPE_RSA')=='y'
assert all(new.get('CONFIG_MODULE_SIG_KEY_TYPE_MLDSA_'+n,'n')!='y' for n in ['44','65','87'])
allowed.add('CONFIG_OPENSSL_SUPPORTS_ML_DSA')
meaningful={k for k in changes if set(changes[k])!={None,'n'}}
assert meaningful<=allowed,('Configuration needs review',{k:changes[k] for k in meaningful-allowed})
# Read only the DAXFS module from the preserved secondary bootstrap archive.
path=PREP/'secondary-bootstrap-daxfs.ko'
with path.open('wb') as out:
 subprocess.run(['cpio','-i','--to-stdout','lib/modules/'+RELEASE+'/daxfs.ko'],input=gzip.decompress((BASE/'artifacts/secondary-daxfs-bootstrap-initramfs.cpio.gz').read_bytes()),stdout=out,stderr=subprocess.PIPE,check=True)
primary=[Path(p) for p in before['preserved_files'] if p.endswith('.ko') and '/rpc-p40-root/' not in p]
secondary=list((BASE/'rpc-p40-root/lib/modules'/RELEASE).glob('*.ko'))+[path]
newsyms=symvers(BUILD/'Module.symvers')
first=next(BUILD.rglob('*.ko'))
vermagic=subprocess.check_output(['modinfo','-F','vermagic',str(first)],text=True).strip()
report={'release':RELEASE,'candidate_vermagic':vermagic,'config_changes':changes,'groups':{},'passed':False}
for group,modules in [('primary',primary),('secondary',secondary)]:
 providers={}
 for module in modules:
  for name,crc in mod_data(module,'--show-exports').items():providers.setdefault(name,set()).add(crc)
 checked=0;external=set();bad=[]
 for module in modules:
  vm=subprocess.check_output(['modinfo','-F','vermagic',str(module)],text=True).strip()
  if vm!=vermagic:bad.append({'module':str(module),'vermagic':vm})
  for name,crc in mod_data(module,'--show-modversions').items():
   checked+=1
   if name in newsyms:
    if crc!=newsyms[name]:bad.append({'module':str(module),'symbol':name,'module_crc':crc,'candidate_crc':newsyms[name]})
   elif crc in providers.get(name,set()):external.add(name)
   else:bad.append({'module':str(module),'symbol':name,'module_crc':crc,'candidate_crc':None})
 report['groups'][group]={'module_count':len(modules),'imports_checked':checked,'unchanged_module_provider_symbols':sorted(external),'mismatches':bad}
 (PREP/'abi-report.json').write_text(json.dumps(report,indent=2)+'\n')
 assert not bad,(group,bad[:10])
report['passed']=True
report['candidate_image_sha256']=hashlib.sha256((BUILD/'arch/x86/boot/bzImage').read_bytes()).hexdigest()
report['preserved_module_files_verified']=len(before['preserved_files'])
(PREP/'abi-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='groups'},indent=2))
for group,value in report['groups'].items():print(group,'modules',value['module_count'],'imports',value['imports_checked'],'mismatches',len(value['mismatches']))
