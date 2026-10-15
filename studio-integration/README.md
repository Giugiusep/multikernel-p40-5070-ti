# Existing Unsloth Studio integration

Studio: http://100.81.184.58:8888 (private Tailscale address). Existing login/username `unsloth` and password are unchanged.

In Chat, select the saved provider **Shiba local models (Strata / Multikernel / RTX)** and choose one of its 19 model profiles. Switching models uses the existing single-resident model API on loopback19080; Strata, RTX-only and optimized Multikernel launch settings remain owned by that API. Model context limits stay as previously configured. Presence in the selector means the model files/profile are available, not that all 19 were freshly benchmarked.

`/home/shiba/models` is also registered as a native Studio scan folder. Native loading uses Studio's existing local llama-server with CUDA0; it does not inherit the custom RPC/Strata profile. Use the saved provider for the established configurations. TTS/STT endpoints remain on the model API; this registration does not add live microphone or audio playback to Studio's frontend.

The existing installation and built frontend are reused without updating or reinstalling packages. Studio runs as enabled user unit `shiba-unsloth-studio.service`, binds only to its Tailscale address, has no Cloudflare tunnel, and retains authentication. Its native llama-server override points to the installed `.unsloth/llama.cpp/build/bin/llama-server`, avoiding old distributed launch wrappers. No model starts automatically inside Studio.

`configure_studio.py` registers available profiles and encrypted credentials using Studio's installed storage helpers. It reads the existing API key locally without printing it, verifies round-trip decryption, and makes a private SQLite backup (including WAL content) under `.unsloth/studio/` before changing settings. No databases/credentials/backups are committed.

Recovery:

1. Restore the existing model API and model files.
2. Run `/home/shiba/.unsloth/studio/unsloth_studio/bin/python /home/shiba/multikernel-experiment/studio-integration/configure_studio.py`.
3. Install the user unit from `configs/systemd/shiba-unsloth-studio.service`, then `systemctl --user daemon-reload` and `systemctl --user enable --now shiba-unsloth-studio.service`.

Validation: Studio `/` returned HTTP200; auth status reports existing initialized account; all19 saved IDs match the provider's `/models` response; Studio's own `ExternalProviderClient` decrypted the saved key and generated “Hello!” through Strata. See registration.json and validation.json. Existing NVIDIA/kernel/Multikernel/TTS/STT configurations were not modified.
