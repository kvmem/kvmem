# Radiance Windows / Linux AMD backend

Build and run with `tools/build.py --platform auto|windows|linux` and
`tools/serve.py --platform auto|windows|linux` on the target OS.

The adapter implements native page ownership, Host spill/restore, recurrent
checkpoints, prefix reuse and rolling generation. It uses mean-key relevance
statistics and the common KVMem pressure picker. Windows includes GPU-address
width fixes and its dedicated memory allocation/mapping path.

Default profile: RX9060XT/gfx1200, BUDGET36864, generation reserve16384,
batch512, context262144, FP8 K/V, MTP3. Service: localhost18380, model radiance-local.

- [Build and deployment](../../docs/radiance-integration.md)
- [Client examples](../../docs/radiance/CLIENT-QUICKSTART.md)
- [Performance results](../../docs/radiance/PERFORMANCE-COMPARISON.md)
- [WSL runtime preparation](wsl-runtime/README.md)

Pinned source and patch metadata are in upstream.json. Preserve upstream
LICENSE/NOTICE with redistributed sources and binaries.
