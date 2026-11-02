from pathlib import Path
from collections import Counter
import csv,json,math
root=Path(__file__).resolve().parent

def read(name,expected):
    rows=list(csv.DictReader(s for s in (root/name).read_text().splitlines() if not s.startswith('#')))
    assert len(rows)==expected
    for r in rows:
        assert 0<float(r['median_us'])<=float(r['p95_us'])
        assert 0<=float(r['relative_L1'])<.03
        m=int(r['M']);assert int(r['entries'])==10*m
        assert int(r['groups'])==(10 if r['routing']=='shared' else 10*m)
    return rows
normal=read('p40-expert-speed-run2.csv',96)
iq1=read('iq1-normal-speed2.csv',32)
read('p40-expert-speed-run1.csv',96);read('iq1-normal-speed1.csv',32)
rows=normal+iq1
layers=[s.split() for s in Path('/home/shiba/multikernel-experiment/research/Strata/packs/iq2_xs/native_experts.txt').read_text().splitlines() if s and not s.startswith('#')]
counts=Counter(int(s[1]) for s in layers);assert counts=={22:34,16:11,29:3}
summary={'description':'Actual native_expert_grouped mode8; synthetic weights, H2560 FF640; gate/up, SwiGLU, internal activation quantization and down; excludes input quantization, Hadamard preprocessing, routing-weighted combine, attention, dense layers and RPC. Not model tok/s.','experts_per_row':10,'M':list(range(1,9)),'samples':50,'calls_per_batch':16,'p95_definition':'p95 of 16-call batch averages; not individual-call p95','checkpoint_expert_layers':dict(counts),'accepted_rows':rows,'IQ1_M_fixture_failure':'Original deterministic fixture had extremely small weights. Independent CPU check finds max intermediate 1.6122446571e-8, max Q8_1 scale1.26948398197e-10, which rounds to zero in FP16. Raw mode0/8 discrepancy retained; legal normal-scale IQ1_M fixture passes all checks. No kernel patch or production config change.','full_model_tokens_per_second_measured':False}
(root/'expert-speed-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['| Format | Layers | M1 ms | M4 shared ms | M8 shared ms | M8 distinct ms |','|---|---:|---:|---:|---:|---:|']
for ty,name in [(22,'IQ2_S'),(16,'IQ2_XXS'),(29,'IQ1_M normalized')]:
    values=[]
    for m,routing in [(1,'shared'),(4,'shared'),(8,'shared'),(8,'distinct')]:
        r=next(r for r in rows if int(r['gu_type'])==ty and int(r['M'])==m and r['routing']==routing and r['mode']=='graph')
        values.append(float(r['median_us'])/1000)
    lines.append('| '+name+' | '+str(counts[ty])+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
(root/'expert-speed-results.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
