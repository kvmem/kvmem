# llama.cpp LoRA

The llama.cpp service can load one matching GGUF LoRA adapter at startup. The
adapter and its scale apply to every target inference lane for the process lifetime.

In the console, select llama.cpp and open Templates & vision. Enter the adapter
path and optional scale. The default scale is 1. Clearing the path starts the base
model. Change the path or scale by restarting the service.

Native invocation:

```sh
llama-kvmem-server -m model.gguf --lora adapter.gguf --lora-scale 0.75
```

Windows structured launcher parameters are `-Lora PATH -LoraScale 0.75`.
Portable profiles use `Lora` and `LoraScale` in their `parameters` object.
Zero and negative finite scales are accepted. Scales are rounded to float precision,
including underflow to zero. The adapter must match the model architecture, target
tensors and base weights. Hugging Face adapters can be
converted with the pinned backend's `convert_lora_to_gguf.py`.

The service reports the loaded path and scale in `/props.lora` and its startup
configuration. Startup fails on unreadable or incompatible adapters. Its
model-free capability response advertises `fixed_lora: true`, allowing the console
to reject older workers before launch.

The request's model state is fixed. A request containing a `lora` override receives
HTTP 400 on Chat Completions and Responses. Activated LoRA requires invocation
handling and is rejected at startup.

Adapters are owned by the shared base model and outlive all lane contexts.
Existing prefix, RAM and temporary SSD caches belong to that fixed process
configuration. MTP/DFlash draft contexts keep their existing weights and the
adapted target verifies proposed tokens. Acceptance and numerical behavior depend
on the adapter and model.

Functional validation uses synthetic rank-2 adapters on Qwen3.5-0.8B Q8_0 and
RTX 5050 Laptop. It covers zero scale, a visible adapter effect, comparison to
native llama memory, two active lanes, prefix reuse, and MTP3. This qualifies the
integration path; trained-adapter quality and other model/quantization combinations
need their own validation.
