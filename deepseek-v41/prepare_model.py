import json,pathlib,subprocess,shutil
base=pathlib.Path('/home/shiba/multikernel-experiment');source=base/'research/Strata-dsv41-experiment';shared=pathlib.Path('/home/shiba/models/deepseek-v41-shared');model=pathlib.Path('/home/shiba/models/deepseek-v4.1-flash-1017-iq2-xs');pack=model/'strata-pack'
report=json.loads((shared/'provenance.json').read_text());assert len(report['files'])==4 and all(f['verified'] for f in report['files'])
command=[str(base/'research/dsv41-venv/bin/python'),str(source/'tools/dsv41/pack_gguf.py'),'--model',str(model),'--out',str(pack),'--engram',str(shared/'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf'),'--engram-meta',str(shared/'DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf'),'--dspark-noise-token','128799','--dspark-gguf',str(shared/'dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf'),'--mmproj',str(shared/'mmproj-BF16.gguf')]
with (base/'deepseek-v41/packing.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
profile=source/'data/dsv41/profile-2.25.bin';shutil.copyfile(profile,pack/'expert-profile.bin')
exe=json.loads((base/'deepseek-v41/engine-install.json').read_text())['executable']
with (base/'deepseek-v41/pack-info.log').open('w') as log:subprocess.run([exe,'--pack',str(pack),'--pack-info'],check=True,stdout=log,stderr=subprocess.STDOUT)
p=base/'deepseek-v41/strata-deepseek-iq2-xs.json';config=json.loads(p.read_text());config['args'] += ([] if '--expert-profile' in config['args'] else ['--expert-profile',str(pack/'expert-profile.bin'),'--profile-readonly']);p.write_text(json.dumps(config,indent=2)+'\n')
print('Model pack and all indexed files validated.')
