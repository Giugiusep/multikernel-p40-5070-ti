# Public model API via Tailscale Funnel

Base URL: https://shiba2th-1.tail0e03fa.ts.net:8443/v1
Chat: POST /v1/chat/completions. Model aliases/options unchanged; all19 chat profiles available. Existing TTS/STT endpoints also use this HTTPS origin.

Original owner key remains at model-api/api-key. Independent second-user key is model-api/api-keys/user-2.key (directory0700, file0600). Read it locally to share through your preferred private channel; never commit it. Both keys use Authorization: Bearer or X-API-Key and currently have the same API access, not isolated per-user models or quotas. Existing single-model serialization and switching apply to both users. Extra *.key files reload on each authenticated request; deleting one revokes it without restarting/unloading a model. Existing owner key remains startup-loaded.

Enable/recover: tailscale funnel --bg --https=8443 --yes http://127.0.0.1:19080
Disable only this endpoint: tailscale funnel --https=8443 off

The preexisting443 Funnel to127.0.0.1:18081 is retained. Studio stays private at100.81.184.58:8888. Public /health still returns basic API status as before; model catalog/inference/speech/transcription require a valid key.

Validation: local and HTTPS Funnel /v1/models both returned200 for each real key (19models),401 with no/wrong key. A disposable test credential was accepted, deleted and immediately rejected401. Credentials never printed or committed. validate_funnel.py repeats this test, including temporary-key revocation. Host-origin HTTPS verification was performed; an independent off-tailnet client was not available for a separate external-network test.
