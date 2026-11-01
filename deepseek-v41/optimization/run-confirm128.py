import pathlib,subprocess,sys,time,urllib.error
sys.path.insert(0,'/home/shiba/multikernel-experiment/model-api');from validate_tts_api import request
ROOT=pathlib.Path('/home/shiba/multikernel-experiment/deepseek-v41/optimization');py='/home/shiba/multikernel-experiment/research/dsv41-venv/bin/python'
try:
 request('/v1/unload',{})
 subprocess.run(['systemctl','--user','stop','shiba-model-api.service'],check=True)
 for job in ['confirm128_draft']:
  subprocess.run(['systemd-run','--user','--unit=shiba-dsv41-'+job.replace('_','-'),'--wait','--pipe','--collect','--property=MemoryMax=42G','--property=MemoryHigh=38G','--property=MemorySwapMax=0',py,'-u',str(ROOT/(job+'.py'))],check=True)
finally:subprocess.run(['systemctl','--user','start','shiba-model-api.service'],check=True)
