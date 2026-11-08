import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=Path('/home/shiba/multikernel-experiment')
sys.path.insert(0,str(BASE/'model-api'));from validate_tts_api import request
CFG=json.loads((BASE/'deepseek-v41/strata-deepseek-iq2-xs-dspark.json').read_text())
PROMPTS=json.loads((BASE/'deepseek-v41/optimization/prompts.json').read_text())
prev=json.loads((ROOT.parent/'results.json').read_text())['runs'];ref=next(r['tokens'] for r in prev if r['mode']=='off' and r['prompt']=='prose')
ids=PROMPTS['prose']+ref[:8]
report={'purpose':'forced same prefix, window1, n0: logits correctness only; not ordinary decode speed','ids':ids,'runs':[]}
def save():(ROOT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
def main():
 state={'restore_model':json.loads(request('/health')).get('active_model')};(ROOT/'state.json').write_text(json.dumps(state));request('/v1/unload',{});subprocess.run(['systemctl','--user','stop','shiba-model-api.service'],check=True)
 outputs=[]
 for tag,mode in [('off-a','off'),('checked','checked-replace'),('off-b','off')]:
  env=os.environ.copy();env.update(CFG['env']);env.update(DS4_MK_EXPERT=mode,DS4_MK_CACHE_MIB='128',DS4_MK_TRACE='1')
  args=list(CFG['args'])+['--ids',','.join(map(str,ids)),'--window','1','--n','0','--dump-logits',str(ROOT/(tag+'.f32')),'--temperature','0','--seed','42']
  with (ROOT/(tag+'.stderr')).open('w') as err:
   p=subprocess.run([str(BASE/'research/deepseek-decode-build/strata-dsv41'),*args],stdout=subprocess.DEVNULL,stderr=err,env=env,timeout=600)
  assert p.returncode==0
  text=(ROOT/(tag+'.stderr')).read_text();assert 'mk expert disabled' not in text
  a=np.fromfile(ROOT/(tag+'.f32'),dtype='<f4').reshape(len(ids),-1);assert np.isfinite(a).all();outputs.append(a)
  row={'tag':tag,'mode':mode,'shape':list(a.shape),'top_last':np.argsort(a[-1])[-5:][::-1].tolist(),'top_logits_last':np.sort(a[-1])[-5:][::-1].tolist()};report['runs'].append(row);save();print(row,flush=True)
 a,b,c=outputs;diff=np.abs(a-b);top=np.argsort(a[-1])[-2:][::-1];report.update(control_repeat_exact=bool(np.array_equal(a,c)),max_abs_logit_difference=float(diff.max()),last_row_max_abs_difference=float(diff[-1].max()),last_row_top2_margin=float(a[-1,top[0]]-a[-1,top[1]]),top1_agreement_by_position=(a.argmax(1)==b.argmax(1)).tolist(),last_top_ids=top.tolist(),last_top_logits_reference=a[-1,top].tolist(),last_top_logits_remote=b[-1,top].tolist());save();print({k:v for k,v in report.items() if k not in ['ids','runs']},flush=True)
if __name__=='__main__':main()
