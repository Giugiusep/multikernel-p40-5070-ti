"""Read-only, whitelisted model-file bridge over Multikernel VSOCK. No TCP listener."""
import errno,json,os,pathlib,socket,stat,struct,time
if os.environ.get('MODEL_BRIDGE_RETEST_AFTER_HANG') != '1':
    raise SystemExit('Model bridge disabled after 2026-11-02 kernel lockup; do not rerun without diagnosis.')
ROOT=pathlib.Path('/home/shiba/multikernel-experiment/research/strata-p40-20261102')
EXPORTS={'/pack':pathlib.Path('/home/shiba/multikernel-experiment/research/Strata/packs/iq2_xs'),'/weights':pathlib.Path('/home/shiba/models/qwen3.8-flash-next-gsq-rco/IQ2_XS'),'/mtp':pathlib.Path('/home/shiba/multikernel-experiment/research/Strata/mtp/rt'),'/profile.bin':pathlib.Path('/home/shiba/multikernel-experiment/research/Strata/data/expert-profile.bin')}
REQ=struct.Struct('<IIQQ');RESP=struct.Struct('<iIQQ');stats={'requests':0,'read_bytes':0,'read_seconds':0.0,'started':time.time()}
def exact(sock,n):
    data=bytearray()
    while len(data)<n:
        x=sock.recv(n-len(data))
        if not x:raise EOFError
        data.extend(x)
    return bytes(data)
def send(sock,data):
    for i in range(0,len(data),32768):
        view=memoryview(data)[i:i+32768]
        while view:
            try:n=sock.send(view)
            except InterruptedError:continue
            except OSError as e:
                if e.errno not in (errno.ENOSPC,errno.EAGAIN):raise
                time.sleep(.0001);continue
            if not n:raise EOFError
            view=view[n:]
        if i+32768<len(data):time.sleep(.0001)
def resolve(path):
    for alias,base in EXPORTS.items():
        if path==alias:return base
        if base.is_dir() and path.startswith(alias+'/'):
            target=(base/path[len(alias)+1:]).resolve();target.relative_to(base.resolve());return target
    raise FileNotFoundError(errno.ENOENT,'not exported')
def serve(peer):
    while True:
        op,n,off,count=REQ.unpack(exact(peer,REQ.size))
        if n>4096 or count>1048576:raise ValueError('request bounds')
        path=exact(peer,n).decode();payload=b'';mode=0;size=0;status=0
        try:
            if path=='/':mode=stat.S_IFDIR|0o555;names=sorted(k[1:] for k in EXPORTS)
            else:
                p=resolve(path);st=p.stat();mode=st.st_mode;size=st.st_size
                names=sorted(x.name for x in p.iterdir()) if p.is_dir() else []
            if op==2:
                if not stat.S_ISDIR(mode):raise NotADirectoryError
                payload=('\n'.join(names)+'\n').encode()
            elif op==3:
                if path=='/' or not stat.S_ISREG(mode):raise IsADirectoryError
                t=time.monotonic();fd=os.open(p,os.O_RDONLY)
                try:payload=os.pread(fd,count,off)
                finally:os.close(fd)
                stats['read_seconds']+=time.monotonic()-t;stats['read_bytes']+=len(payload)
            elif op!=1:raise OSError(errno.ENOSYS,'operation')
        except (OSError,ValueError) as e:status=-(getattr(e,'errno',None) or errno.EACCES)
        send(peer,RESP.pack(status,mode,size,len(payload)))
        if payload:send(peer,payload)
        stats['requests']+=1
        if stats['requests']%1000==0:(ROOT/'file-server-stats.json').write_text(json.dumps(stats,indent=2)+'\n')
s=socket.socket(socket.AF_VSOCK,socket.SOCK_STREAM);s.setsockopt(socket.AF_VSOCK,9,1);s.bind((socket.VMADDR_CID_ANY,6004));s.listen(4);print('MODEL_FILES_READY port6004 read-only',flush=True)
while True:
    peer,addr=s.accept()
    try:serve(peer)
    except (EOFError,OSError,ValueError,UnicodeError) as e:print(type(e).__name__,str(e),flush=True)
    finally:peer.close();(ROOT/'file-server-stats.json').write_text(json.dumps(stats,indent=2)+'\n')
