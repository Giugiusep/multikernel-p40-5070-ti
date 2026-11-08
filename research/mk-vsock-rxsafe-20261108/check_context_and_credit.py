"""Static callback check plus a deterministic model of unchanged stream credits.
This is not a live-kernel proof or a performance test.
"""
from pathlib import Path
import random,json
root=Path(__file__).resolve().parent
source=(root/'mk_transport.c').read_text();callback=source[source.index('static void mk_vsock_ipi_handler('):source.index('\n/*\n * Transport operations')]
assert 'mk_vsock_rx_pkt(' not in callback and callback.count('mk_queue_rx_pkt(skb);')==1
assert source.count('mk_vsock_rx_pkt(skb);')==1  # receive work item only
rng=random.Random(20261108);mask=(1<<32)-1;window=64*1024
# Start near wrap and delay peer credit advertisements on purpose.
tx=fwd=peer=(1<<32)-100000;wire=rx=0;sent=0;peak=0
for step in range(1200000):
 space=max(window-((tx-peer)&mask),0);n=min(space,rng.randint(1,32768));tx=(tx+n)&mask;wire+=n;sent+=n
 if rng.randrange(3):n=rng.randint(0,wire);wire-=n;rx+=n
 if rx:n=rng.randint(0,rx);rx-=n;fwd=(fwd+n)&mask
 if rng.randrange(3):peer=fwd
 outstanding=(tx-peer)&mask
 assert 0<=wire+rx<=outstanding<=window
 peak=max(peak,wire+rx)
assert sent>1<<32
result={'static_hardirq_callback_queue_only':True,'model_iterations':step+1,'model_payload_bytes':sent,'max_unconsumed_payload_bytes':peak,'per_stream_credit_bytes':window,'limits':'model checks an unchanged credit algorithm, not live queue memory, control floods, lockup root cause or throughput'}
(root/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
