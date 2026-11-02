"""Keep legal synthetic IQ1_M scales away from downstream half-scale underflow."""
from pathlib import Path
import sys,struct,json,hashlib
import numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,'/home/shiba/llama.cpp/gguf-py')
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
source=root/'speed-fixtures/gu-IQ1_M.bin';data=source.read_bytes();ty,rows,cols=struct.unpack('<iii',data[:12]);assert ty==29
raw=np.frombuffer(data[12:],dtype=np.uint8).copy().reshape(-1,56)
scales=raw[:,48:56].copy().view('<u2').reshape(-1,4)
d=np.array([.01],dtype='<f2').view('<u2')[0]
for i in range(4):scales[:,i]=(scales[:,i]&0x0fff)|(((d>>(4*i))&15)<<12)
raw[:,48:56]=scales.view(np.uint8).reshape(-1,8)
ref=dequantize(raw.reshape(-1),GGMLQuantizationType.IQ1_M).reshape(rows,cols).astype('<f4')
assert np.isfinite(ref).all() and np.sqrt(np.mean(ref*ref))>.01
prefix=root/'speed-fixtures/gu-IQ1_M-normal'
prefix.with_suffix('.bin').write_bytes(data[:12]+raw.tobytes());prefix.with_suffix('.f32').write_bytes(ref.tobytes())
(root/'iq1-normal-fixture.json').write_text(json.dumps({'packed_scale':float(np.float16(.01)),'reference_RMS':float(np.sqrt(np.mean(ref*ref))),'original_reference_RMS':float(np.sqrt(np.mean(np.fromfile(root/'speed-fixtures/gu-IQ1_M.f32',dtype='<f4')**2))),'sha256':{suffix:hashlib.sha256(prefix.with_suffix(suffix).read_bytes()).hexdigest() for suffix in ['.bin','.f32']}},indent=2)+'\n')
print('Normalized legal block scale:',float(np.float16(.01)),'RMS',float(np.sqrt(np.mean(ref*ref))))
