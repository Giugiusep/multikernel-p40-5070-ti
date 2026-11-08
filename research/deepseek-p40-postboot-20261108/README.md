# Post-reboot DeepSeek/P40 integration checkpoint

Primary source commit `4465dad`, branch `experiment/shiba-deepseek-decode-release`. Experimental, default off. No production API alias/config/executable changed. This is partial expert compute on one distributed model, not a second model or drafter server.

Restored unchanged kernel7.0.0-mk2-shiba-mk1 and saved secondary rpc-p40:CPUs8–11,1792MiB,P40 passthrough. Driver580.173.02 verified by nvidia-smi. RTX primary CUDA exercised by full-model runs. No driver/package/kernel updates, no additional reboot, no physical host changes. After final tests conservative5000, saved fast5002 and private6011 are responsive. Fast library restored from original archive; SHA2565c0fb8e5279b4c1270f209e08f4d0a154cf95fcc77bd8300c4e41a4986bb75b5. Secondary fast process uses original GGML_MK_VSOCK_NO_READ_SLEEP1/GGML_MK_VSOCK_NO_WRITE_SLEEP1; primary upload pacing untouched. API health ok, no active model (same pre-test state). Public Funnel configuration unchanged; no external reachability test.

## Full-expert path remains unqualified

128MiB admission, trace-enabled16token diagnostic: code off2.58,shadow3.30,checked-replace3.20,actual3.14tok/s; all16tokens exact. Prose off2.51,shadow3.07exact;checked replacement2.16first diverged at token8 despite no worker failure. Max live expert relative-L1 .000409286889. Timing instrumentation and short outputs make these correctness diagnostics, not controlled performance gains. Slot-identity fix alone does not establish full-expert correctness.

Forced identical22token prose prefix,window1,n0,full logits: two CPU controls bit-identical. CPU top token42666/logit37.710403,runner-up10428/36.805527,margin.904877. Checked P40 substitution selects10428/38.132591 while42666falls37.490028. Max logit difference2.367229 across all rows;last row1.668874. This is NOT merely a near-tie. Amplification through routing/recurrent/cache quantization is a hypothesis, not a located first divergent layer. Original full-expert replacement must not be enabled in API. Binary logit arrays retained locally, excluded from git.

## Byte-exact down-only alternative

Keep CPU gate/up, CPU SwiGLU/libm and actual CPU Q8_K quantization. Offload only down projections of cached IQ2_XXS/IQ2_XS/IQ2_S experts. GPU emulates CPU's eight integer accumulation groups, FP32 per-block multiply/FMA, high/low reduction order and final1/8 factor. No shared dequantized model, Tensor Cores or SM70 instructions. Widths1–8,hidden5120/FFN2304. Other formats/geometries/cache misses remain CPU. All six routed experts, shared expert, combine order, Engram and DSpark unchanged.

Mode DS4_MK_EXPERT=down-shadow runs CPU and GPU and requires raw output bytes equal;down skips the corresponding CPU down jobs. DS4_MK_CACHE_MIB sets bounded16–1024MiB client admission;worker1GiB hard limit/16MiBframes. No eviction/upload churn. Full three-role blobs are still uploaded to this prototype cache; GPU uses only down weights. Host input Q8_K is unpacked losslessly into int8 codes and replicated signed scales. AF_VSOCK type1 socket selection,32KiB/100us sends and partial I/O loops unchanged. On worker error the adapter disables itself and retains CPU fallback. These modes are private; production remains original release binary.

128MiB real-engine gate:36code plus26prose expert comparisons have zero differing output bits. Actual offload matches32+32committed tokens. Independent pinned ggml-cpu oracle: real checkpoint down weights for each of3supported formats x widths1–8 =24cases; outputs bit-identical. Final timing validation repeats each24times,576byte-exact results. This is evidence for these shapes/formats/inputs, not a general proof for arbitrary checkpoints or contexts.

First GPU implementation decoded floats then divided back to integer coefficients. IQ2_XXS M1:median2.197070ms roundtrip/1.981440msGPU. Direct packed decode: .727072ms/.529408ms. Four rows per warp, preserving8lanes/row: .449649ms/.243712ms.20timed samples after3warmups,plus initial checked call; final p95.468244ms. Synthetic roundtrip4.886x faster; NOT an inference speedup claim. Other widths/types and rejected baseline sources retained.

## Full-model performance gate

Same DeepSeek setup as preceding tests; Nemotron was not benchmarked: IQ2_XS pack,context8192,CPU8,cache32GBdecimal,all6experts,DSpark5/conf.65,greedy seed42,fixed21/14token code/prose prompts.64generated tokens per arm. CLI rate=(N-1)/decode_ms. ABBA off/down/down/off,2repeats per prompt.42GiB hard/38GiBsoft,no swap,core dumps disabled.512MiBP40admission;detailed traces off on both sides. No compiling,hashing or console transfers during timing. Each campaign has512committed tokens exact to preceding CPU reference,zero reported worker failure. No NVIDIA/kernel changes.

| Candidate | Prompt | RTX+CPU controls tok/s | P40 down offload tok/s | Tokens |
|---|---|---|---|---|
| First exact kernel | code | 5.09 / 4.67 | 3.17 / 3.54 | exact |
| First exact kernel | prose | 2.72 / 4.22 | 3.63 / 4.27 | exact |
| Packed / fuller warp | code | 4.68 / 4.74 | 3.98 / 4.56 | exact |
| Packed / fuller warp | prose | 2.56 / 4.48 | 4.37 / 3.4 | exact |

No consistent full-model improvement; do not deploy either candidate as a speedup. Packed decoder improves isolated kernel latency and code offload versus the first exact prototype, but code still trails same-window controls. Prose varies broadly, including large control changes. Do not attribute every model delta to GPU compute or transport from these aggregate timers.

512MiB runs admit58experts/529182720bytes. Code shows170completed down requests/228total requests,534532632application-protocol bytes including cold registrations,prompt and decode. Request counters are whole-run, NOT per generated token. Primary StoreHold still reads complete expert blobs before down offload, so this experiment saves CPU down jobs but DOES NOT eliminate SSD reads or reduce primary store footprint. Raw code store reads remain393.8MB/internal token. Down-only serialization,misses and redundant weight traffic are next engineering targets; avoid more naive kernel search without profiling full-model wall time.

## Transport warning and recovery limits

Secondary dmesg WARNING at126.081857s:__local_bh_enable_ip in hardirq,stack vsock_find_connected_socket_net→mk_vsock_rx_pkt→mk_vsock_ipi_handler. Static source confirms RW packets are handled inside IPI;socket table uses spin_lock_bh,not IRQ disabling. That is an interrupt-context locking hazard and explains the warning condition. It is NOT proof of the previous complete secondary stall's exact cause, and no new kernel patch was installed. After all tests dmesg still ends at126.082222s, no newer ring-full/Xid/reset lines in the captured tail;P40memory301MiBbeforefastRPCrestoration. Kernel warning is real and must not be hidden under a claim of clean dmesg. Running inside KVM may affect timing/PCI behavior, but no evidence identifies outer KVM as this warning's cause.

Saved diagnostic worker6011 stays available, original5000andrestored5002unchanged. No secondary kill/reset/module reload. CUDA builds for sm120 and sm61 passed; HIP and Metal were not built. Source patch,both reference kernels,oracle,resource output,all raw token sequences and benchmark JSON included;executables,objectfiles,weights and logit arrays excluded from backup.
