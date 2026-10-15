"""Exercise Studio's installed provider client with its saved encrypted credential."""
import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0,'/home/shiba/.unsloth/studio/unsloth_studio/lib/python3.13/site-packages/studio/backend')
from storage import credential_secrets,providers_db
from core.inference.external_provider import ExternalProviderClient

async def main():
    provider=providers_db.get_provider('shiba-local-models')
    client=ExternalProviderClient(provider['provider_type'],provider['base_url'],
                                  credential_secrets.get_provider_api_key(provider['id']),timeout=600)
    try:
        models=await client.list_models()
        ids={m['id'] for m in models}
        assert set(provider['models'])<=ids
        result=await client.chat_completion([{'role':'user','content':'Reply with a short greeting.'}],
                   model='qwen3.8-flash-next-strata',temperature=0,max_tokens=96)
        message=result['choices'][0]['message']
        assert message.get('content'),message
        output={'provider_id':provider['id'],'catalog_count':len(ids),'registered_count':len(provider['models']),
                'model':result.get('model'),'answer':message['content'],'saved_credential_used':True}
        Path('/home/shiba/multikernel-experiment/studio-integration/validation.json').write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps(output,indent=2))
    finally:await client.close()

asyncio.run(main())
