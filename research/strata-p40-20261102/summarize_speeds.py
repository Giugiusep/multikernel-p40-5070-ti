"""Validate measurements and summarize warmed runs; no inference-throughput estimate."""
from pathlib import Path
import csv,json,math,re,statistics
root=Path(__file__).resolve().parent
runs=[]
for i in range(1,5):
    rows=list(csv.DictReader(s for s in (root/f'p40-speed-run{i}.csv').read_text().splitlines() if not s.startswith('#')))
    assert len(rows)==64
    indexed={}
    for r in rows:
        key=(Path(r['case']).name,int(r['M']),r['mode']);assert key not in indexed
        med,p95=float(r['median_us']),float(r['p95_us']);assert 0<med<=p95
        assert float(r['relative_L1'])<.02
        assert int(r['copies'])==64 and int(r['samples'])==50
        assert int(r['weight_bytes'])*64>4*3145728
        assert math.isclose(float(r['effective_weight_GBs']),int(r['weight_bytes'])/med/1000,rel_tol=1e-5)
        indexed[key]=r
    runs.append(indexed)
assert all(set(r)==set(runs[0]) for r in runs)
summary=[]
for key in sorted(runs[2]):
    a,b=runs[2][key],runs[3][key]
    med=statistics.median([float(a['median_us']),float(b['median_us'])])
    summary.append({'case':key[0],'M':key[1],'mode':key[2],'median_of_warmed_run_medians_us':med,'max_warmed_batch_p95_us':max(float(a['p95_us']),float(b['p95_us'])),'run3_us':float(a['median_us']),'run4_us':float(b['median_us']),'nominal_weight_GBs':int(a['weight_bytes'])/med/1000,'relative_L1':float(a['relative_L1'])})
drift=max(abs(float(runs[3][k]['median_us'])/float(runs[2][k]['median_us'])-1)*100 for k in runs[2])
assert drift<2,drift
text=(root/'speed-secondary-dmesg.log').read_text();ring=[float(t) for t in re.findall(r'\[\s*([\d.]+)\] multikernel: IPI ring full',text)]
gpu=list(csv.DictReader((root/'speed-gpu.csv').open()));gpu=[{k.strip():v.strip() for k,v in r.items()} for r in gpu]
result={'description':'Synthetic packed matrices at real Qwen shapes, GPU matvec only; not model tok/s','warmed_runs':[3,4],'maximum_warmed_median_drift_percent':drift,'batch_size':64,'batch_samples':50,'p95_definition':'p95 of 64-call batch averages, maximum of two warmed runs; not individual-call p95','rows':summary,'max_sampled_SM_MHz':max(int(r['clocks.current.sm [MHz]'].split()[0]) for r in gpu),'max_sampled_temperature_C':max(int(r['temperature.gpu']) for r in gpu),'ring_full_reports':len(ring),'ring_full_time_span':[min(ring),max(ring)] if ring else None,'Xid_reports':len(re.findall('NVRM: Xid',text)),'OOM_reports':len(re.findall('Out of memory:|oom-kill:',text))}
(root/'speed-summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['| Matrix | M=1 µs | M=2 µs | M=4 µs | M=8 µs |','|---|---:|---:|---:|---:|']
for case in ['gu-IQ2_XXS','gu-IQ2_XS','gu-IQ2_S','down-Q2_0']:
    vals=[next(r['median_of_warmed_run_medians_us'] for r in summary if r['case']==case and r['M']==m and r['mode']=='graph') for m in [1,2,4,8]]
    lines.append('| '+case+' | '+' | '.join(f'{x:.2f}' for x in vals)+' |')
(root/'speed-results.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines));print('Maximum warmed median drift:',drift,'%')
