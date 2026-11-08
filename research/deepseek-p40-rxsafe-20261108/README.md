# RX-safe postboot DeepSeek correctness gate


### Full-model postboot gate completed

All six 32-token DeepSeek runs passed exact token agreement, without worker disable/timeouts. Code CPU/off 3.30, compact-shadow 4.83, compact-P40 4.17 tok/s; prose CPU/off 3.41, compact-shadow 3.68, compact-P40 2.69 tok/s. Shadow comparisons:424 code and414 prose, all exact. These single sequential trials have storage/cache confounding and do NOT establish a P40 performance win. Keep production settings unchanged. Peak cgroup memory35.8GiB; API restored with health ok and no active model. Both kernels remained responsive. Post-test primary kernel search and secondary dmesg found no new ring-full, RCU stalls or Xids. Outstanding: concurrent-listener reset and live worker-loss/lazy-slot fallback. Persistent GRUB default remains old #9, candidate was one-shot only.

See results.json and validation.log for exact worker and end-to-end evidence.
