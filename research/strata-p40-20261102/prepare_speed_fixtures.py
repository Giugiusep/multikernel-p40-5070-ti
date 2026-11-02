from pathlib import Path
import json,hashlib,shutil,struct,sys
import numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,'/home/shiba/multikernel-experiment/research/Strata-dsv41-experiment/tools')
from gguf_writer import quantize_q2_0,dequantize_q2_0
out=root/'speed-fixtures';out.mkdir(exist_ok=True)
for name in ['IQ2_XXS','IQ2_XS','IQ2_S','IQ1_M']:
    for suffix in ['bin','f32']:shutil.copy2(root/'iq-fixture-qwen-gu'/(name+'.'+suffix),out/('gu-'+name+'.'+suffix))
rows,cols=2560,640
rng=np.random.RandomState(17);matrix=rng.standard_normal((rows,cols)).astype(np.float32)*np.float32(.02)
raw=quantize_q2_0(matrix);reference=dequantize_q2_0(raw).reshape(rows,cols).astype('<f4')
assert reference.shape==(rows,cols) and np.isfinite(reference).all()
(out/'down-Q2_0.bin').write_bytes(struct.pack('<iii',42,rows,cols)+raw)
(out/'down-Q2_0.f32').write_bytes(reference.tobytes())
(root/'speed-fixture-hashes.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir()},indent=2)+'\n')
print('Three Qwen gate/up formats and Q2_0 down at actual matrix dimensions; synthetic weights.')
