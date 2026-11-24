"""Evaluate the prepared integration's CMake dependency declarations without a GPU compiler.

This catches missing target edges; native CI must still compile/link the targets.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SOURCE = Path(os.environ.get("POLYSTRATA_SOURCE", Path(__file__).resolve().parents[2]/"backends/polystrata"))
RECORDER = r'''
cmake_minimum_required(VERSION 3.24)
function(add_executable name)
  set_property(GLOBAL PROPERTY "${name}_sources" "${ARGN}")
endfunction()
function(add_library name)
  set_property(GLOBAL PROPERTY "${name}_sources" "${ARGN}")
endfunction()
function(target_sources name)
  set_property(GLOBAL APPEND PROPERTY "${name}_sources" "${ARGN}")
endfunction()
function(target_link_libraries name)
  set_property(GLOBAL APPEND PROPERTY "${name}_links" "${ARGN}")
endfunction()
function(get_target_property variable target property)
  if(NOT property STREQUAL "SOURCES")
    message(FATAL_ERROR "Unsupported target property: ${property}")
  endif()
  get_property(value GLOBAL PROPERTY "${target}_sources")
  set(${variable} "${value}" PARENT_SCOPE)
endfunction()
function(target_include_directories)
endfunction()
function(target_compile_definitions)
endfunction()
function(set_source_files_properties)
endfunction()
function(set_tests_properties)
endfunction()
function(add_test)
endfunction()
function(enable_testing)
endfunction()
function(add_subdirectory)
  add_library(kvmem STATIC portable_core)
endfunction()
function(kvmem_enable_cuda_recall)
  # Portable core behavior is outside the integration block being evaluated.
endfunction()
'''


class BuildContract(unittest.TestCase):
    def graph(self, backend):
        cmake = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")
        if not cmake:
            self.skipTest("CMake is required to evaluate target declarations")
        source = (SOURCE/"engine/cmake/strata_poly.cmake").read_text(encoding="utf-8")
        begin = source.index("  if(STRATA_ENABLE_CUDA AND TARGET strata_gdn_mma)\n    target_compile_definitions(strata_poly_mmq PUBLIC STRATA_POLY_GDN_MMA=1)")
        end = source.index("  # needs the model file, so it is run by hand: poly_expert_parity", begin)
        with tempfile.TemporaryDirectory(prefix="poly cmake contract ") as directory:
            root = Path(directory)
            (root/"include/kvmem").mkdir(parents=True)
            (root/"include/kvmem/memory_contract.hpp").touch()
            script = RECORDER + f'\nset(STRATA_KVMEM_SOURCE_DIR "{root.as_posix()}")\n'
            script += f'set(STRATA_ENABLE_CUDA {"ON" if backend == "cuda" else "OFF"})\n'
            script += f'set(STRATA_ENABLE_HIP {"ON" if backend == "hip" else "OFF"})\n'
            script += 'set(_strata_gpu_runtime_target gpu_runtime)\nset(_strata_gpu_blas_target gpu_blas)\n'
            script += source[begin:end].replace("STRATA_ENABLE_CUDA AND TARGET strata_gdn_mma", "STRATA_ENABLE_CUDA") + '''
foreach(t strata-poly kvmem kvmem_recall_hip poly_gdn_mma_state_test poly_elastic_test poly_kvmem_mtp_test)
  foreach(p sources links)
    get_property(v GLOBAL PROPERTY "${t}_${p}")
    message("CONTRACT|${t}|${p}|${v}")
  endforeach()
endforeach()
'''
            path = root/"contract.cmake"
            path.write_text(script, encoding="utf-8")
            result = subprocess.run([cmake, "-P", str(path)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            graph = {}
            for line in (result.stdout+result.stderr).splitlines():
                if line.startswith("CONTRACT|"):
                    _, target, kind, values = line.split("|", 3)
                    graph.setdefault(target, {})[kind] = values.split(";") if values else []
            return graph

    def test_cuda_state_target_keeps_driver_link(self):
        graph = self.graph("cuda")
        self.assertIn("CUDA::cuda_driver", graph["poly_gdn_mma_state_test"]["links"])
        for target in ("strata-poly", "poly_elastic_test", "poly_kvmem_mtp_test"):
            self.assertIn("CUDA::cuda_driver", graph[target]["links"])

    def test_hip_scorer_is_owned_by_the_shared_core(self):
        graph = self.graph("hip")
        self.assertIn("$<TARGET_OBJECTS:kvmem_recall_hip>", graph["kvmem"]["sources"])
        self.assertIn("hip::host", graph["kvmem"]["links"])
        self.assertIn("gpu_runtime", graph["kvmem_recall_hip"]["links"])
        self.assertNotIn("$<TARGET_OBJECTS:kvmem_recall_hip>", graph["strata-poly"]["sources"])
        for target in ("strata-poly", "poly_elastic_test", "poly_kvmem_mtp_test"):
            self.assertIn("kvmem", graph[target]["links"])
            self.assertNotIn("CUDA::cuda_driver", graph[target]["links"])
            self.assertIn("src/poly/kolibri1.cpp", graph[target]["sources"])


if __name__ == "__main__":
    unittest.main()
