import json,os,pathlib,re,subprocess,time,sys
ROOT=pathlib.Path('/home/shiba/multikernel-experiment/deepseek-v41/optimization');cfg=json.load(open('/home/shiba/multikernel-experiment/deepseek-v41/strata-deepseek-iq2-xs.json'));prompts=json.load(open(ROOT/'prompts.json'));rows=[]
variants=[('fixed8',8),('fixed12',12),('fixed4',4)]
if len(sys.argv)>1:variants=[v for v in variants if v[0] in sys.argv[1:]]
for label,threads in variants:
 args=list(cfg['args']);args[args.index('--cpu-threads')+1]=str(threads);args+=['--no-expert-cache','--ids',','.join(map(str,prompts['code'])),'--n','64','--temperature','0','--seed','42']
 env=os.environ.copy();env.update({'DS4_TOKLOG':'1','DS4_ROUND_TIMES':'1','DS4_IQ_VERBOSE':'1','STRATA_DS4_DISK_SKIP_W':'0'})
 start=time.monotonic()
 with (ROOT/(label+'.stderr')).open('w') as err,(ROOT/(label+'.tokens')).open('w') as out:
  completed=subprocess.run([cfg['exe'],*args],stdout=out,stderr=err,env=env,timeout=900)
 text=(ROOT/(label+'.stderr')).read_text();tokens=[int(v) for v in (ROOT/(label+'.tokens')).read_text().strip().strip(',').split(',') if v.strip()]
 row={'variant':label,'threads':threads,'exit':completed.returncode,'total_seconds':time.monotonic()-start,'ids':tokens,'metrics':[line for line in text.splitlines() if any(k in line for k in ['decode:','decode per token:','store:','token latency','lookahead:','rounds:','CPU lane IQ'])]};rows.append(row);(ROOT/'cli-profile.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
 if completed.returncode:raise RuntimeError(label+' failed')
 if len(rows)>1 and tokens!=rows[0]['ids']:print('OUTPUT MISMATCH: '+label,flush=True)
