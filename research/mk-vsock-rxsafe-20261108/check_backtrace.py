"""Verify the observed interrupted instruction lies inside the socket-table lock."""
import re,json
from pathlib import Path
root=Path(__file__).resolve().parent
trace=(root/'live-rcu-stall-3.log').read_text();asm=(root/'vsock-table-disassembly.log').read_text()
assert '<IRQ>' in trace and '<TASK>' in trace
assert 'vsock_find_connected_socket_net+0x2c/0x210' in trace
assert 'RIP: 0010:vsock_find_connected_socket_net+0x67/0x210' in trace
base=int(re.search(r'([0-9a-f]+) <vsock_find_connected_socket_net>:',asm)[1],16)
lock=int(re.search(r'^([0-9a-f]+):.*call.*<_raw_spin_lock_bh>',asm,re.M)[1],16)
unlock=int(re.search(r'^([0-9a-f]+):.*call.*<_raw_spin_unlock_bh>',asm,re.M)[1],16)
assert lock+5==base+0x2c and lock+5<base+0x67<unlock
assert 'ffffffff83c11720' in trace and 'ffffffff83c11720' in asm
print('PASS: IRQ blocks acquiring the same socket-table lock held by the interrupted task on CPU0')
