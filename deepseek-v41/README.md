# Isolated DeepSeek V4.1 Flash setup

User-selected AtomicChat IQ2_XS, original4shard payload kept in models/deepseek-v4.1-flash-1017-iq2-xs. Required/shared Engram, metadata, optional DSpark and vision weights in models/deepseek-v41-shared. Provenance JSONs and download manifests record revisions, byte counts and independently verified SHA256 hashes.

Engine is upstream strata-dsv410.1.55 build99866a47a from signedv0.2.9 Linuxx86_64CUDA13 package. Archive SHA2568eed51bf333bc3bc381bc3cc4daa7dbca51d612647732e6f2a63c63be4c66124, package signature and all component files verified through upstream install_package. Runtime CUDA13 libraries are next to the executable; no system CUDA toolkit or NVIDIA package replacement. Python dependencies live in research/dsv41-venv and are recorded in python-requirements.txt. Source checkout research/Strata-dsv41-experiment is pinnedc62d5e3a15b40ea51c96aae9e8d36928965f53ab; deployed Qwen Strata tree is separate.

prepare_model.py generates an index-only strata-pack with explicit Engram metadata, drafter and vision references. No duplicate model payload is created. Initial config is full6expert,8K context,8CPU threads,32GBRAM expert cache,1GiBVRAM reserve, DSparknone and visionoff. Drafter/vision are installed for later tests, not claimed validated by their download. start-isolated.sh uses loopback19085; run it with a systemd user scope/service limiting memory to42G, high38G, swap0. Unload the shared API model first and ensure no native Studio model uses RTX.

Recovery: restore source checkout and venv; download engine archive from https://github.com/Niko1221/Strata/releases/download/v0.2.9/strata-dsv41-0.1.55-linux-x86_64-cuda-cuda13.tar.gz into research/deepseek-v41-plan; run install_engine.py; restore model files using model-download manifests; run prepare_model.py; run the isolated service then validate_local.py before API exposure. Do not run both the isolated server and API-managed model simultaneously.

## Accepted initial serving profile

API model ID: `deepseek-v4.1-flash-1017-uncensored-iq2-xs`.
Public base: https://shiba2th-1.tail0e03fa.ts.net:8443/v1
Existing API keys work. Studio's saved Shiba provider has20models, including this alias. API launches the new upstream server through a scoped42Gmemory limit/high38G/swap0; old Qwen commands remain unchanged. Tests cover both launch paths and GPU ownership diagnostics.

Validated: all companion sizes/hashes, signed engine/all package files, every indexed pack file, narrow deterministic Paris/42answers, a coherent64token paragraph, old Qwen→DeepSeek switching and second-user public streaming with DONE.64token paragraph:5.22decode tok/s (engine timer),15.01send-to-complete seconds including2.74s prompt work; not a sustained throughput benchmark. Initial API load+short answer22.58s; subsequent switch-back+greeting28.32s. Peak experimental service memory38.18GB; final API scope peak40.80GB, no OOM/max events and zero scope swap. Global VM swap may still be used by other processes.

Not established: long-context quality, long-run reliability, DSpark correctness/performance, image input, native audio on this new backend, P40/Multikernel integration. Initial config is8K,6experts,no drafter/vision. No full-model equivalence or task-quality claim follows from these smoke checks. Separate existing TTS/STT endpoints remain available; the DeepSeek chat alias does not advertise integrated speech. About383GiB disk remains free after installation.

API request example:

```bash
curl https://shiba2th-1.tail0e03fa.ts.net:8443/v1/chat/completions \
 -H "Authorization: Bearer $SHIBA_MODEL_API_KEY" -H 'Content-Type: application/json' \
 -d '{"model":"deepseek-v4.1-flash-1017-uncensored-iq2-xs","messages":[{"role":"user","content":"Explain how a bicycle works."}],"max_tokens":128,"reasoning_effort":"none","stream":false}'
```

Recovery of serving config: restore API models.json/model_switch_api.py and this folder; install pinned Python requirements into research/dsv41-venv, restore source/engine/files, run prepare_model.py, then restart shiba-model-api.service. Do not start the old isolated test service alongside the API model. No driver/kernel update or reboot is part of recovery.
