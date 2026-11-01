import json,pathlib,os,subprocess,sys,time,urllib.request,urllib.error
ROOT=pathlib.Path('/home/shiba/multikernel-experiment/deepseek-v41/optimization');SOURCE=pathlib.Path('/home/shiba/multikernel-experiment/research/Strata-dsv41-experiment');sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(SOURCE/'tools'))
import strata_tokenizer as ST
from serve.server import Service,MockEngine
from serve.frontend import ChatTemplate,openai_to_messages
from serve import dsml
cfg=json.load(open('/home/shiba/multikernel-experiment/deepseek-v41/strata-deepseek-iq2-xs.json'));tp=pathlib.Path(cfg['tokenizer']);v=json.loads((tp/'vocab.json').read_text());tokens=[None]*len(v)
for t,i in v.items():tokens[i]=t
meta=json.loads((tp/'tokenizer.json').read_text());tok=ST.Tokenizer(tokens,(tp/'merges.txt').read_text().split('\n'),json.loads((tp/'token_type.json').read_text()),meta['pre'],meta['special_ids']);tpl,_=dsml.pick_template(tp/'chat_template.jinja');svc=Service(MockEngine(tok,''),tok,ChatTemplate(tpl));svc.model_arch='deepseek41';svc.set_family('deepseek41')
prompts={}
for label,text in [('prose','Explain how a bicycle works in one clear paragraph.'),('code','Write a Python function that returns the first n Fibonacci numbers. Include a brief explanation.')]:
 req={'messages':[{'role':'user','content':text}],'reasoning_effort':'none','max_tokens':32};messages,tools,kw=openai_to_messages(req);ids,thinking,n=svc.prepare(messages,tools,kw,32);prompts[label]=ids
(ROOT/'prompts.json').write_text(json.dumps(prompts,indent=2)+'\n');print('prompt token counts', {k:len(v) for k,v in prompts.items()},flush=True)
report=json.loads((ROOT/'repeatability.json').read_text()) if (ROOT/'repeatability.json').exists() else {}
for mode,extra in [('adaptive',[]),('fixed',['--no-expert-cache'])]:
 if len(sys.argv)>1 and mode not in sys.argv[1:]:continue
 env=os.environ.copy();env.update({'STRATA_DS4_REUSE':'0','STRATA_DS4_SLOT_KEEP':'0','STRATA_DS4_AUTO_PIN':'0','DS4_ROUND_TIMES':'1'})
 args=[cfg['exe'],*cfg['args'],*extra,'--serve'];log=(ROOT/(mode+'-raw-engine.log')).open('w')
 p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env,cwd=SOURCE)
 try:
  while True:
   line=p.stdout.readline()
   if not line:raise RuntimeError(mode+' startup failed')
   if line.startswith('READY'):break
  rows=[]
  for repeat in range(2):
   for label,ids in prompts.items():
    command='GEN 32 temperature=0 top_p=1 top_k=1 seed=42 '+','.join(map(str,ids))+'\n';p.stdin.write(command);p.stdin.flush();start=time.monotonic();out=[]
    while True:
     line=p.stdout.readline()
     if not line:raise RuntimeError('engine stopped')
     if line.startswith('T '):out.append(int(line.split()[1]))
     if line.startswith('ERR'):raise RuntimeError(line)
     if line.startswith('DONE '):done=line.strip();break
    row={'prompt':label,'repeat':repeat,'ids':out,'text':tok.decode(out),'done':done,'wall_seconds':time.monotonic()-start};rows.append(row);print(mode,label,repeat,done,flush=True)
    report[mode]={'rows':rows};(ROOT/'repeatability.json').write_text(json.dumps(report,indent=2)+'\n')
  report[mode]['exact_repeats']={label:next(r['ids'] for r in rows if r['prompt']==label and r['repeat']==0)==next(r['ids'] for r in rows if r['prompt']==label and r['repeat']==1) for label in prompts}
 finally:
  if p.poll() is None:
   p.stdin.write('QUIT\n');p.stdin.flush();p.stdin.close();p.wait(timeout=30)
  log.close()
 (ROOT/'repeatability.json').write_text(json.dumps(report,indent=2)+'\n');print(mode,report[mode]['exact_repeats'],flush=True)
