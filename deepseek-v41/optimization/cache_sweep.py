import json,os,pathlib,subprocess,time
ROOT=pathlib.Path('/home/shiba/multikernel-experiment/deepseek-v41/optimization');cfg=json.load(open('/home/shiba/multikernel-experiment/deepseek-v41/strata-deepseek-iq2-xs.json'));prompts=json.load(open(ROOT/'prompts.json'));reference=json.load(open(ROOT/'cli-profile.json'))[0]['ids'];report=[]
for label,policy,gb in [('lru32','lru',32),('lfu32','lfu',32),('lfulru32','lfulru',32),('lru36','lru',36)]:
 args=list(cfg['args']);args[args.index('--ssd-cache')+1]=str(gb);args+=['--no-expert-cache','--ssd-policy',policy,'--ids',','.join(map(str,prompts['code'])),'--n','64','--temperature','0','--seed','42']
 env=os.environ.copy();env.update({'DS4_TOKLOG':'1','STRATA_DS4_DISK_SKIP_W':'0'})
 with (ROOT/(label+'.stderr')).open('w') as err,(ROOT/(label+'.tokens')).open('w') as out:
  result=subprocess.run([cfg['exe'],*args],stdout=out,stderr=err,env=env,timeout=900)
 text=(ROOT/(label+'.stderr')).read_text();tokens=[int(x) for x in (ROOT/(label+'.tokens')).read_text().strip().strip(',').split(',') if x.strip()]
 row={'label':label,'policy':policy,'cache_GB':gb,'exit':result.returncode,'exact_agreement':tokens==reference,'ids':tokens,'metrics':[x for x in text.splitlines() if any(k in x for k in ['decode:','decode per token:','store:','token latency'])]};report.append(row);(ROOT/'cache-sweep.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
 if result.returncode or not row['exact_agreement']:raise RuntimeError(label+' failed correctness gate')
