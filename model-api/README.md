# Switchable model API

`shiba-model-api.service` listens on port 19080 and exposes an authenticated OpenAI-compatible chat endpoint. It advertises the installed language GGUFs in `models.json`, loads the requested model on demand, and unloads the previous model before switching. Model backends bind only to `127.0.0.1:19081`. The API service binds to the VM's network interfaces; it requires the key stored in `api-key` (mode 0600).

On this VM, the API is reachable at `http://192.168.178.73:19080/v1` on the LAN and `http://100.81.184.58:19080/v1` on Tailscale. Both addresses were checked on 2026-09-29. Use the bearer key from the local `api-key` file; the key is not printed in logs or documentation.

This service does not manage Hermes or Unsloth Studio. The original `nemotron-3.5-lightning-q8` entry is a local RTX/CPU fallback that works without the P40 worker. The separate `nemotron-3.5-lightning-q8-mk` entry uses the RTX and the P40 over native Multikernel VSOCK. MiMo, Mistral, and the other untested models remain on their local runners; the API switches between these profiles by model ID.

```bash
systemctl --user status shiba-model-api.service
export SHIBA_MODEL_API_KEY="$(cat /home/shiba/multikernel-experiment/model-api/api-key)"

curl -H "Authorization: Bearer $SHIBA_MODEL_API_KEY" \
  http://127.0.0.1:19080/v1/models

curl -H "Authorization: Bearer $SHIBA_MODEL_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"mimo-v2.6-q6-k"}' \
  http://127.0.0.1:19080/v1/switch

curl -H "Authorization: Bearer $SHIBA_MODEL_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"mimo-v2.6-q6-k","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}' \
  http://127.0.0.1:19080/v1/chat/completions

curl -H "Authorization: Bearer $SHIBA_MODEL_API_KEY" \
  -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:19080/v1/unload
```

For OpenAI-compatible clients, use `http://<VM-IP>:19080/v1` as the base URL, the contents of `api-key` as the API key, and one of the IDs returned by `/v1/models` as the model. A `model` is required on every generation request. Both non-streaming and SSE streaming chat completions are forwarded. Strata also accepts `/v1/messages`; its backend does not implement the legacy text-completions endpoint.

The `available` field means a model file is present, not that its output has passed a correctness test. The original 13 catalog entries loaded and answered a simple deterministic text prompt through this API on 2026-09-29; see `model-smoke-results.json`. The fourteenth, distributed Nemotron alias, was isolated-probed at 160K before being added. These are serving smoke tests, not model-quality or full-context validation. Muse Glimmer needed a 320-token output budget to finish its reasoning and return `4`; its first 128-token attempt ended with an empty final answer and `finish_reason=length`. Multimodal projector files and standalone draft/MTP/DFlash files are not separate chat models in this catalog, and vision input has not been configured or tested.

K2 uses the publisher's separate `model/K2Horizon` fork because the current upstream llama.cpp build does not recognize its GGUF architecture. Strata uses the isolated diagnostic build with PCIe expert execution and GPU cache hits bypassed: the previous default path reproduced a k=4 token mismatch. Keep its output labeled experimental until broader correctness testing passes. The currently exposed Strata profile uses its isolated RTX engine, not the native llama.cpp Qwen4exp MTP branch. The original local Nemotron 3.5 profile uses RTX/CPU and was very slow in the smoke test; select the `-mk` alias for the validated distributed placement.

Cold model switches can take minutes. Only one request is served at a time, including while a streaming response is open; this prevents a switch from unloading weights mid-request. Backend logs are in `logs/`. The key file and logs are ignored by this directory's `.gitignore`.

The API context allocations were raised after isolated startup probes on this 49 GiB RAM / 16 GiB RTX VM. They are token counts, including the prompt and generated output:

| Model ID | API context |
|---|---:|
| `qwen3.8-flash-next-strata` | 262,144 |
| `ling-3-flash-iq2-s` | 262,144 |
| `mimo-v2.6-q6-k` | 262,144 |
| `k2-horizon-mova-q4-k-m` | 262,144 |
| `nemotron3-nano-4b-q4-k-m` | 917,504 |
| `nemotron-3.5-lightning-q8` | 1,048,576 |
| `nemotron-3.5-lightning-q8-mk` | 160,000 |
| `mistral-small-4-iq2-xxs` | 524,288 |
| `qwen3.8-27b-q8`, `qwen3.8-27b-q4-xl` | 262,144 |
| `qwen3.8-27b-iq2-s-rtx` | 180,000 |
| `gemma-4-e4b-q2-xl` | 131,072 |
| `gemma-4-26b-iq3-xxs` | 262,144 |
| `glm-4.7-flash-q6-xl` | 202,752 |
| `muse-glimmer-30b-q8` | 131,072 |

K2 needs Q8 KV and eight GPU layers at 262K. Nano needs Q8 KV and a 512-token prefill batch at 896K. Mistral retains F16 KV and four GPU layers at 524K; its 1M startup requires CPU-only execution and Q8 KV, so it is not the default. Distributed Nemotron uses Q8 KV, a 128-token batch and a 44/56 RTX/P40 layer split at 160K. These checks prove startup and short-response behavior only; filling the whole context can expose additional memory pressure and may be slow.

`qwen3.8-27b-iq2-s-rtx` was downloaded from `unsloth/Qwen3.8-27B-GGUF` and added as a CUDA0-only profile with every layer offloaded to the RTX 5070 Ti, Q8 KV, flash attention, batch 512 and ubatch 128. Its GGUF SHA-256 is `7897d2c5a5cee46aef50895141b2c8a0803c1185f3d03c4fda4cd137a7ad77fe`. A 262,144-token allocation failed when the KV cache requested another 8.7 GiB; the configured 180,000-token context loaded (llama.cpp rounded it to 180,224) and left 843 MiB free on the 16 GiB RTX. Three 256-token greedy generations measured a **66.76 tok/s median decode rate**. The benchmark's 1,353-token prompt had 42 cached tokens and processed the remaining 1,311 at a **1,341.73 tok/s median**. These are short-live-context speed measurements with a 180K allocation, not a test at 180K live tokens. Results: `qwen3.8-27b-iq2-s-rtx-benchmark-180000.json`; the 128K trial is in `qwen3.8-27b-iq2-s-rtx-benchmark-131072.json`.

After the context change, API chat was repeated successfully on the changed Strata, K2, Nano, Mistral and local Nemotron 3.5 profiles; see `high-context-smoke-results.json`. The other eight entries passed isolated startup at their new context and had passed chat at the earlier smaller setting. No full-window prompt test has been run.

To repeat a sequential smoke test, run `python3 smoke_models.py <model-id> [<model-id> ...]`; use `--max-tokens 320` for models that spend much of a small completion budget on reasoning. The script records responses and timings in `model-smoke-results.json`. Both Strata and llama-server SSE chat streaming were checked through the public API.

## Multikernel model probes and API choice

The isolated `probe_context.py --rpc` path and the `-mk` API alias use the native patched llama.cpp coordinator and the P40 worker on `mkvsock:1:5002`. Primary write pacing remains enabled, and the conservative RPC port 5000 remains available. The original local model IDs are unchanged.

The live API switch to this alias succeeded in 860.27 seconds. Six subsequent 128-token greedy API chats matched the isolated probe's response hash; decode rates ranged from 57.992 to 58.570 tok/s. `python3 smoke_multikernel_api.py` repeats the active-model, catalog and response check without reloading the resident model. The old `nemotron-api.service`, which still pointed at an obsolete TCP RPC address, was disabled from autostart; the P40 RPC workers themselves were not altered.

At 160K allocated context, the 50/50 Nemotron Q8 split ran out of RTX VRAM during startup. The 44/56 split loaded in 860.3 seconds and returned three identical 128-token greedy responses, matching the earlier 4K output SHA-256 `b57e083d…5370f177`. Its two warmed decode rates were 57.969 and 58.184 tok/s on a short live prompt. This is the configuration offered as `nemotron-3.5-lightning-q8-mk`; loading it cold takes roughly 14 minutes, and switching to any other model unloads it. It has not processed a prompt that fills 160K.

At 4K allocated context on a 256-token greedy Mistral Small 4 prompt, the RTX/P40 38/62 split returned exactly the same output hash as local execution and measured 48.02 and 48.04 decode tok/s in two warmed runs. The current local Unsloth backend had a seven-run median of 24.29 tok/s on the same prompt, with substantial run-to-run variation. Mistral's distributed cold load took about 16 minutes. GLM 4.7 Flash Q6_XL measured 45.88 tok/s distributed versus 9.33 tok/s on the patched RTX/RAM coordinator for one 128-token prompt; their measured outputs matched, and its distributed cold load took 11.2 minutes. These are limited short-context comparisons.

**Do not reuse the 131,072-token Mistral split profile yet.** Although it loaded and completed one 256-token response at 47.01 tok/s, the VM froze during the next response and its last saved kernel event was a `kcompactd0` soft lockup. The user manually shut down the VM. No 131K result was written to `context-probe-results.json`, and no distributed Mistral alias was published. After reboot, the P40 secondary was restored and the validated Nemotron profile was tested separately. Details are in `multikernel-engineering-audit.md`.


### HauhauCS RTX-only profile (2026-10-07)

Model ID: `qwen3.8-27b-hauhau-iq2-m-rtx`. Requested context60,000 (actual slot60,160), Q8KV, all model layers onCUDA0, noRPC. Verified model-file SHA256 and three deterministic chat outputs; approximately59.7decode tok/s on a short prompt. Ordinary decoding is active; embeddedMTP tensors are not enabled. The existing endpoint and bearer key are unchanged. Select this ID through `/v1/switch` or `/v1/chat/completions`; other models remain available.
