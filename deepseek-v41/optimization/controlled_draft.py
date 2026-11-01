import json,pathlib,subprocess,os,time
ROOT=pathlib.Path('/home/shiba/multikernel-experiment/deepseek-v41/optimization');cfg=json.load(open('/home/shiba/multikernel-experiment/deepseek-v41/strata-deepseek-iq2-xs.json'));prompts=json.load(open(ROOT/'prompts.json'));args=list(cfg['args']);args[args.index('--dspark')+1]='on';args+=['--no-expert-cache','--dspark-conf','0.65','--dspark-block','3','--serve'];env=os.environ.copy();env.update({'STRATA_DS4_REUSE':'0','STRATA_DS4_SLOT_KEEP':'0','STRATA_DS4_AUTO_PIN':'0','DS4_ROUND_TIMES':'1','STRATA_DS4_DISK_SKIP_W':'0'});report={'rows':[]}
log=(ROOT/'controlled-draft.stderr').open('w');p=subprocess.Popen([cfg['exe'],*args],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
try:
 while True:
  line=p.stdout.readline()
  if not line:raise RuntimeError('load failed')
  if line.startswith('READY'):break
 for repeat in range(2):
  for label,ids in prompts.items():
   for mode,spec in [('loaded-off',0),('loaded-on',1)]:
    p.stdin.write('GEN 32 temperature=0 top_p=1 top_k=1 seed=42 spec='+str(spec)+' '+','.join(map(str,ids))+'\n');p.stdin.flush();out=[];start=time.monotonic()
    while True:
     line=p.stdout.readline()
     if not line:raise RuntimeError('engine stopped')
     if line.startswith('T '):out.append(int(line.split()[1]))
     if line.startswith('ERR'):raise RuntimeError(line)
     if line.startswith('DONE '):done=line.strip();break
    row={'mode':mode,'prompt':label,'repeat':repeat,'ids':out,'done':done,'wall_seconds':time.monotonic()-start};report['rows'].append(row)
    if spec:
     ref=next(r for r in report['rows'] if r['mode']=='loaded-off' and r['prompt']==label and r['repeat']==repeat);row['exact_target_agreement']=out==ref['ids']
     if not row['exact_target_agreement']:row['first_different_token']=next((i for i,(a,b) in enumerate(zip(out,ref['ids'])) if a!=b),min(len(out),len(ref['ids'])))
    (ROOT/'controlled-draft.json').write_text(json.dumps(report,indent=2)+'\n');print(mode,label,repeat,done,'match',row.get('exact_target_agreement'),flush=True)
finally:
 if p.poll() is None:p.stdin.write('QUIT\n');p.stdin.flush();p.stdin.close();p.wait(timeout=30)
 log.close()
