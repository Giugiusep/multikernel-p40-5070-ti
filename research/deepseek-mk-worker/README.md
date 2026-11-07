# Bounded DeepSeek/P40 expert worker

Experimental port6010, separate from Qwen6005–6008 and GGML RPC5000/5002. No API integration, filesystem bridge, NVIDIA changes or Multikernel changes. Primary retains model/Engram/DSpark ownership. This is a resident expert service and validation harness, not a complete DeepSeek server.

## Contract and limits

AF_VSOCK with explicit setsockopt(AF_VSOCK,9,1). Both sends retain32768-byte pieces and100us progress pacing, partial I/O loops, EINTR handling, and bounded worker ENOSPC/EAGAIN retries. Ten-second socket deadlines. Little-endian x86-64 protocol, magic0x4458454d/version1; six uint32 frame fields:magic,version,operation,status,sequence,body bytes. Frame<=16MiB. Each connection owns its cache, freed on disconnect. One peer at a time. GPU cache hard limit1GiB; hidden5120,FFN2304,40layers,384experts,1–8 input rows. No network model misses or filesystem paths accepted.

* INFO0: empty request; JSON contract deepseek-iq-q8-32-v1.
* LOAD1: layer,expert,gate/up type,down type,hidden,FFN,one gate matrix bytes,down matrix bytes (eight uint32), followed by gate,up,down checkpoint-native bytes. Only observed types accepted; exact body lengths, shapes, IDs and duplicate registration checked. Gate/up share a format; down independently typed. No implicit format substitution.
* RUN2: layer,expert,rows,debug (four uint32), then int8 input[rows][5120] and FP32 scales[rows][160]. Exact byte count, finite nonnegative scales, code range and residency checked. One resident expert evaluated over all rows, not a general multi-expert routing batch. Internal entry stride8 matches DeepSeek grouped-kernel semantics. Gate/up and down use existing ds4_iq kernels, intermediate Q8 uses maximum absolute value per32, nearest-even codes. SwiGLU limit10. Return four uint32 rows/debug/hidden/FFN,three FP32 CUDA-event durations in microseconds,and FP32 outputs[rows][5120]. Debug additionally returns FP32 SwiGLU[rows][2304],int8 intermediate codes and FP32 per32 scales. Results are unweighted expert contributions; primary accumulation/routing not implemented here.
* CLEAR3: empty body; free registered experts. Unknown operations and malformed requests return explicit errors; malformed frame terminates connection. CUDA faults close the connection. No public TCP listener.

## Build and run

Existing prerequisite: engine-review-20261107's pinned scalar-only SM61 objects (see its build_probe.sh/manifest). `bash build.sh` links a new private worker and independent ggml decoder. It does not modify a reference archive or serving binary. `dsv41-venv/bin/python validate.py --iterations30` needs an already-running worker at CID1:6010. Never substitute the old Qwen worker protocol. The standalone secondary runtime loader/library paths are the same preserved paths as the previous qualified probes. No source/binary transfer is automated by this script; use the existing paced helper and verify hashes before execution.

## Correctness gate and measurements

Representative expert0 from each of ten actual gate/down format combinations in the existing GGUF pack. Widths1–8:80 cases. Primary extracts only bounded expert blobs from indexed local shards, hashes each role, and decodes them with ggml's independent CPU decoder. Deterministic activations include zero32-value blocks. Float64 CPU dot/SwiGLU references check GU, down and complete expert output; returned GPU intermediate quantization must equal CPU quantization. GPU stage relative-L1 gates:GU5e-5/down2e-5, max normalized errors2e-4. Complete independent CPU-GU/quantize/down gate1e-3, allowing intermediate quantization-boundary sensitivity. These are explicit tolerance gates, NOT byte-exact equivalence to a full model.

Final run passes all80 cases,2640 repeated timing outputs, plus per-row calls requiring raw-byte equality to the grouped outputs, SwiGLU and quantization. Twenty-six short writes occurred and were handled successfully. Maximum GU relative-L1:1.3899e-7;down7.2713e-8;complete expert1.4397e-4. Invalid lengths/layouts/widths/scales, duplicates, unknown operations, missing experts and post-clear misses rejected. Actual1GiB exhaustion and malformed frame disconnect fuzzing not exercised. Cache map supports multiple registrations, but this gate clears between representative experts; general multi-expert routing remains unqualified.

| verification rows | roundtrip ms | GPU us |
|---:|---:|---:|
|1|0.590|376.3|
|2|0.729|358.4|
|3|1.004|526.8|
|4|1.289|630.8|
|5|1.454|632.1|
|6|1.736|686.1|
|7|1.902|693.2|
|8|2.183|795.1|

Each table value is the median across10 per-case medians,30 timed repeats after3 warmups. Per-case p95 and ranges in results.json/summary.json. CUDA events separate GU/intermediate quantization/down; roundtrip includes socket/pacing, CPU handling, metadata and staging. Debug data/reference calculations are excluded from timed calls. Binaries frozen, no concurrent builds. API remained live; these are probe measurements, not isolated full-system/model benchmarks. All rows share one expert; routing diversity can destroy reuse. Wider output also increases socket/pacing costs. No full-model tok/s claim or promotion.

## Next integration gate

Inspected source boundaries: src/dsv41/ds4_gguf.cu moe_gate_up/moe_down and win_experts already quantize inputs into per32 int8/FP32-scale buffers and launch the same IQ kernels. A future opt-in primary adapter should shadow selected resident IDs, capture those actual quantized buffers and expert outputs, and compare before replacing any work. Keep combine order, routing weights, shared experts, all six selected experts, Engram, DSpark and current serving settings unchanged. Cover different experts/rows and cache misses before enabling replacement. Current deployed engine is a preserved release binary; this service alone does not inject an adapter into it. Build/validate a separate primary engine first. No full DeepSeek model runs in the secondary.
