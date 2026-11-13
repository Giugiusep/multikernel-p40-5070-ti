from pathlib import Path
import json,re,statistics
p=Path(__file__).resolve().parent;j=json.loads((p/'results.json').read_text());assert j['passed'];rows=[]
for i,r in enumerate(j['runs'][1:]):
 s=(p/f'down-compact-code-64-{i}.stderr').read_text();cs=[dict((k,float(v)) for k,v in re.findall(r'(\w+)=([\d.]+)',l)) for l in s.splitlines() if l.startswith('mk checkpoint ')]
 assert len(cs)>=2;d={k:cs[-1][k]-cs[0][k] for k in cs[0]};a=re.search(r'CPU lane ([\d.]+).*head wait',s);w=re.findall(r'CPU lane waited ([\d.]+) ms',s)
 rows.append({'cache_mib':r['cache_mib'],'tok_s':r['tps'],'exact':r['exact_agreement'],'loaded_experts':r['loads'],'decode_delta':d,'decode_per_interval':{k:v/63 for k,v in d.items()},'disk_wait_ms_per_token':float(w[-1])})
summary={'rows':rows,'mean_tok_s':{m:statistics.mean(r['tok_s'] for r in rows if r['cache_mib']==m) for m in (512,1024)},'interpretation':'No reliable 1GiB gain in these short storage-confounded ABBA runs; do not promote. RPC counters are cumulative difference after prefill, divided by 63 generated intervals. GPU time is included in RPC, do not add. Disk wait counters overlap CPU/layer wall time.'}
(p/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
