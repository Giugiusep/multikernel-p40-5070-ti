from pathlib import Path
import json,re,statistics
p=Path(__file__).resolve().parent
small=json.loads((p.parent/'deepseek-p40-large-warm-20261113/results.json').read_text());large=json.loads((p.parent/'deepseek-p40-large-warm22-20261113/results.json').read_text());assert all(r['exact'] for r in small['runs']+large['runs'])
summary={'capacity':json.loads((p/'capacity-results.json').read_text()),'cold':[],'warm':{},'note':'Warm requests are identical prompt replay in persistent process with prompt reuse disabled; 63 intervals/decode time, exact64 CPU-reference tokens. Two warm samples per budget; not a broad benchmark.'}
for i,cap in enumerate([512,8192,22528]):
 s=(p/f'down-compact-code-64-{i}.stderr').read_text();cs=[dict((k,float(v)) for k,v in re.findall(r'(\w+)=([\d.]+)',l)) for l in s.splitlines() if l.startswith('mk checkpoint ')];d={k:cs[-1][k]-cs[0][k] for k in cs[0]};ms=re.search(r'mk summary .*resident=(\d+)',s);r=json.loads((p/'results.json').read_text())['runs'][i+1]
 summary['cold'].append({'budget_mib':cap,'actual_weight_bytes':int(ms[1]),'tok_s':r['tps'],'exact':r['exact_agreement'],'decode_delta':d})
for name,obj in [('512',small),('22528',large)]:
 rows=[r for r in obj['runs'] if r['warm']];summary['warm'][name]={'tok_s':[r['tok_s_63_intervals'] for r in rows],'mean_tok_s':statistics.mean(r['tok_s_63_intervals'] for r in rows),'exact':all(r['exact'] for r in rows)}
(p/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='capacity'},indent=2))
