"""Add one separate 1GiB range; never remove/resize the running kernel region."""
from pathlib import Path
import json,struct,subprocess
import libfdt
from kerf.runtime import DeviceTreeManager
ROOT=Path(__file__).resolve().parent
manager=DeviceTreeManager()
def chunks():
    f=libfdt.Fdt(Path('/sys/fs/multikernel/device_tree').read_bytes());out=[];node=f.path_offset('/resources')
    try:c=f.first_subnode(node)
    except libfdt.FdtException:return out
    while c>=0:
        if f.get_name(c).startswith('memory@'):
            out.append(struct.unpack('>QQ',bytes(f.getprop(c,'reg'))))
        try:c=f.next_subnode(c)
        except libfdt.FdtException:break
    return out
def apply(text,name):
    p=ROOT/(name+'.dts');p.write_text(text);out=ROOT/(name+'.dtbo');subprocess.run(['dtc','-q','-I','dts','-O','dtb','-o',str(out),str(p)],check=True);return manager.apply_dtbo(out.read_bytes())
with manager.lock():
    before=set(chunks())
    pool_tx=apply('/dts-v1/; /plugin/; / { fragment@0 { target-path="/resources"; __overlay__ { memory-add { memory@0 { size=/bits/ 64 <1207959552>; numa-node-id=<0>; }; }; }; }; };','hotadd-pool')
    new=set(chunks())-before;assert len(new)==1,new
    base,size=new.pop();align=128*1024**2;start=(base+align-1)//align*align;length=1024**3;assert base<=start and start+length<=base+size
    name=f'hotadd-instance-{start:x}'
    tx=apply(f'/dts-v1/; /plugin/; / {{ fragment@0 {{ target-path="/instances/rpc-p40"; __overlay__ {{ memory-add {{ memory@{start:x} {{ reg=/bits/ 64 <0x{start:x} 0x{length:x}>; numa-node-id=<0>; }}; }}; }}; }}; }};',name)
    result={'pool_transaction':pool_tx,'instance_transaction':tx,'pool_base':base,'pool_bytes':size,'added_base':start,'added_bytes':length,'operation':'add-only; original region preserved'}
    with (ROOT/'hotadd-ram.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    print(json.dumps(result),flush=True)
