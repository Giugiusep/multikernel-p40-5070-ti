"""Install only separate candidate boot files and select one test boot. Never reboot."""
from pathlib import Path
import hashlib,json,os,re,shutil,socket,subprocess
BASE=Path('/home/shiba/multikernel-experiment')
PREP=BASE/'research/mk-vsock-rxsafe-20261108/boot-preparation'
BUILD=BASE/'research/linux-rxsafe-build'
RELEASE='7.0.0-mk2-shiba-mk1'
LABEL=RELEASE+'-rxsafe1'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(*args):return subprocess.check_output(list(args),text=True)
assert os.geteuid()==0 and socket.gethostname()=='shiba2th'
assert run('uname','-r').strip()==RELEASE
report=json.loads((PREP/'abi-report.json').read_text());assert report['passed']
assert sha(BUILD/'arch/x86/boot/bzImage')==report['candidate_image_sha256']
assert 'rxsafe1' in (BUILD/'include/generated/utsversion.h').read_text()
assert (BUILD/'include/config/kernel.release').read_text().strip()==RELEASE
before=json.loads((PREP/'before.json').read_text())
for name,meta in before['preserved_files'].items():assert sha(name)==meta['sha256'],name
persistent=Path('/etc/default/grub.d/99-multikernel-experiment.cfg')
persistent_sha=sha(persistent)
recovery=PREP/'boot-files-before';recovery.mkdir(exist_ok=True)
for p in [Path('/boot/grub/grub.cfg'),Path('/boot/grub/grubenv'),persistent]:
 target=recovery/p.name
 if not target.exists():shutil.copy2(p,target)
files={
 '/boot/vmlinuz-'+LABEL:BUILD/'arch/x86/boot/bzImage',
 '/boot/initrd.img-'+LABEL:Path('/boot/initrd.img-'+RELEASE),
 '/boot/config-'+LABEL:BUILD/'.config',
 '/boot/System.map-'+LABEL:BUILD/'System.map',
}
installed={}
for name,src in files.items():
 target=Path(name);expected=sha(src)
 if target.exists():assert sha(target)==expected,('Existing candidate differs',name)
 else:
  temp=target.with_name(target.name+'.partial')
  assert not temp.exists(),str(temp)
  with src.open('rb') as inp,temp.open('xb') as out:
   shutil.copyfileobj(inp,out,1024*1024);out.flush();os.fsync(out.fileno())
  os.chmod(temp,0o644);assert sha(temp)==expected;os.replace(temp,target)
 installed[name]={'sha256':sha(target),'size':target.stat().st_size}
subprocess.run(['update-grub'],check=True)
subprocess.run(['grub-script-check','/boot/grub/grub.cfg'],check=True)
assert sha(persistent)==persistent_sha,'Persistent boot default changed'
text=Path('/boot/grub/grub.cfg').read_text()
submenu=re.search(r"^submenu .*'([^']*gnulinux-advanced-[^']*)'",text,re.M).group(1)
entry=re.search(r"^\tmenuentry .*'(gnulinux-"+re.escape(LABEL)+r"-advanced-[^']*)'",text,re.M).group(1)
old=re.search(r"^\tmenuentry .*'(gnulinux-"+re.escape(RELEASE)+r"-advanced-[^']*)'",text,re.M).group(1)
assert old in persistent.read_text() and entry!=old
selection=submenu+'>'+entry
subprocess.run(['grub-reboot',selection],check=True)
env=run('grub-editenv','/boot/grub/grubenv','list')
assert 'next_entry='+selection in env.splitlines()
for name,meta in before['preserved_files'].items():assert sha(name)==meta['sha256'],name
assert sha('/boot/initrd.img-'+LABEL)==sha('/boot/initrd.img-'+RELEASE)
result={'installed_boot_files':installed,'one_time_next_entry':selection,'persistent_default_unchanged':True,'preserved_file_count':len(before['preserved_files']),'all_baseline_hashes_unchanged':True,'boot_release':RELEASE,'build_marker':'#12-rxsafe1','secondary_kernel':'/boot/vmlinuz-'+LABEL,'reboot_executed':False,'grub_environment':env.splitlines()}
(PREP/'installed.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print('STOP: ready for user reboot; no reboot or secondary restart executed.')
