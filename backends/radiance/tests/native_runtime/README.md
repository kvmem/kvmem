# Native platform probes

These small platform tests exercise GPU copies, tensor parsers and direct-I/O block logic in the patched Radiance runtime.

Requirements: CMake3.25+, Ninja, a Windows ROCm SDK with Clang/HIP development
files, Visual Studio C++ build tools/Windows SDK, and the patched Radiance source.
Choose the installed SDK explicitly.

PowerShell example (replace the three local paths):

```powershell
$Sdk = 'D:\path\to\rocm'
$Source = 'D:\path\to\patched-radiance'
$Build = 'D:\path\to\new-platform-test-build'
$Targets = @(& "$Sdk\lib\llvm\bin\amdgpu-arch.exe" | Sort-Object -Unique)
if ($LASTEXITCODE -ne 0 -or $Targets.Count -ne 1) {
    throw 'Select a target explicitly for this single-architecture diagnostic build.'
}
cmake -S . -B $Build -G Ninja `
  "-DCMAKE_CXX_COMPILER=$Sdk/lib/llvm/bin/clang++.exe" `
  "-DROCM_ROOT=$Sdk" "-DRADIANCE_SOURCE=$Source" `
  "-DRAD_GPU_TARGET=$($Targets[0])" -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw 'Configure failed' }
cmake --build $Build -j 2
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
$PreviousPath = $env:PATH
try {
    $env:PATH = "$Sdk\bin;$PreviousPath"
    ctest --test-dir $Build --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw 'Platform tests failed; preserve the output' }
} finally {
    $env:PATH = $PreviousPath
}
```

The GPU probe enumerates visible devices and selects the one with most available
VRAM, or accepts an explicit device index as its sole argument. This diagnostic build targets one GPU architecture.

Tests:

- `native_runtime_probe`: production device dispatch,64KiB transfers, cross-stream
  event dependencies,16 ranged-copy kernel launches, and host virtual-memory
  reserve/commit/decommit/recommit.
- `native_module_test`: DLL path with spaces, local symbol lookup, errors/unload.
- `native_file_test`: concurrent positional reads, offsets/mapping boundaries,
  short reads and closed handles.
- `native_tensor_file_test`: SafeTensors, GGUF, legacy imatrix payload values and
  malformed/truncated input through the actual parsers.
- `native_direct_file_test`: actual unbuffered transfers, cross-sector updates,
  short final sector, EOF rejection.

File tests create only named fixtures inside the build directory, refuse an
existing fixture, and remove their own fixture on success. Failure preserves it
for diagnosis. Use a new build/fixture path for a rerun rather than overwriting
the failed evidence.

Freeze shared C++ header edits before compiling Windows and WSL consumers.
Changing a class layout during a running compile can produce mixed-layout
objects whose modification times appear current. Rebuild all affected consumers
before executing tests when that happens.
