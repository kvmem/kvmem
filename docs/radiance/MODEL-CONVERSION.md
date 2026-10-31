# Model conversion provenance

The upstream AMD model is already Quark AWQ MXFP4. The backbone weights are retained.
The local additions use Radiance's existing RTN quantizer.

`backends/radiance/recipes/qwen27-mxfp4-mtp.recipe` is the **baseline** recipe: MTP
projections remain BF16 and the main output head is FP8. The recipe
`qwen27-six-mtp-mxfp4.recipe`, adds the six actual MTP projection rules used in
the first local conversion (FP4 E2M1 / groups32 / E8M0 scale).

The second historical conversion reused the first container but took `output.weight`
from the original checkpoint and overrode only that rule:

```sh
rad-convert /path/to/amd-source --home /path/to/radiance-home \
  --max-spec 3 --kv-cache-dtype fp8 --max-tok 512 --max-ctx 262144 \
  --recipe /path/to/kvmem/backends/radiance/recipes/qwen27-six-mtp-mxfp4.recipe \
  --quant 'output.weight=rtn:codes=fp4_e2m1,group=32,scale=e8m0' \
  --reuse /path/to/mtp-mxfp4.rad -o /path/to/target-head-mxfp4-a.rad
```

For a fresh conversion, use a new output path, run `--plan-only`, and audit complete payloads with the included `compare_rad_target.py` / `compare_mtp_quant.py` tools.
The original audits found 1033 unchanged tensors plus six MTP changes, then 1038
unchanged tensors plus one output-head change. Output-head quantization changes logits
and can change draft acceptance/quality.

The container retains a dedicated U2 MTP output tensor, while the tested deployment
sets `KVMEM_RAD_MTP_HEAD=shared` to use the main head at runtime. That setting is in
the explicit Windows profile. The tested
attention cache uses FP8 K and FP8 V.
