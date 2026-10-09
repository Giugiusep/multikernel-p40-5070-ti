"""Controlled image checks for the separate vision profile."""
import base64,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
KEY=(ROOT/'api-key').read_text().strip()
rows=[]
for file,expected,prompt in [('red.png','red','What is the dominant colour in this image? Reply with just the colour name.'),('blue.png','blue','What is the dominant colour in this image? Reply with just the colour name.'),('three-circles.png','3','How many red circles are in this image? Reply with only the numeral.')]:
 image=base64.b64encode((ROOT/'vision-fixtures'/file).read_bytes()).decode()
 payload={'model':'qwen3.8-flash-next-strata-vision','messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+image}},{'type':'text','text':prompt}]}],'temperature':0,'max_tokens':128}
 request=urllib.request.Request('http://127.0.0.1:19080/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
 start=time.monotonic()
 with urllib.request.urlopen(request,timeout=720) as response:data=json.load(response)
 assert data['choices'][0]['message']['content'].strip().lower()==expected
 rows.append({'expected':expected,'seconds':time.monotonic()-start,'response':data})
(ROOT/'strata-vision-validation.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Three controlled image checks passed')
