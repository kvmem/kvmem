cmake_minimum_required(VERSION 3.25)
project(radiance_native_runtime_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(RADIANCE_SOURCE "" CACHE PATH "Pinned Radiance source with platform patch")
set(ROCM_ROOT "" CACHE PATH "ROCm SDK")
set(RAD_GPU_TARGET "" CACHE STRING "Detected or explicitly selected GPU architecture")
if(NOT EXISTS "${RADIANCE_SOURCE}/core/device/hip_runtime.cpp" OR
   NOT EXISTS "${ROCM_ROOT}/include/hip/hip_runtime.h" OR NOT RAD_GPU_TARGET MATCHES "^gfx[0-9a-f]+$")
  message(FATAL_ERROR "Set RADIANCE_SOURCE, ROCM_ROOT and an explicit/detected RAD_GPU_TARGET")
endif()
find_library(HIP_RUNTIME NAMES amdhip64 PATHS "${ROCM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_path(HIP_DEVICE_LIBS ocml.bc PATHS "${ROCM_ROOT}/lib/llvm/amdgcn/bitcode"
  "${ROCM_ROOT}/amdgcn/bitcode" NO_DEFAULT_PATH REQUIRED)
add_executable(native_runtime_probe runtime_test.cpp
  "${RADIANCE_SOURCE}/core/device/hip.cpp"
  "${RADIANCE_SOURCE}/core/device/hip_runtime.cpp"
  "${RADIANCE_SOURCE}/core/device/dispatch.cpp"
  "${RADIANCE_SOURCE}/core/device/host.cpp"
  "${RADIANCE_SOURCE}/core/util/rad_util.cpp"
  "${RADIANCE_SOURCE}/core/device/aql_blit.hip")
target_include_directories(native_runtime_probe PRIVATE
  "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/core" "${ROCM_ROOT}/include")
target_compile_definitions(native_runtime_probe PRIVATE RAD_HAVE_HIP=1 RAD_USE_HIP_RUNTIME=1 __HIP_PLATFORM_AMD__=1 _CRT_SECURE_NO_WARNINGS)
set_source_files_properties("${RADIANCE_SOURCE}/core/device/aql_blit.hip" PROPERTIES
  LANGUAGE CXX COMPILE_OPTIONS "-x;hip;--rocm-path=${ROCM_ROOT};--rocm-device-lib-path=${HIP_DEVICE_LIBS};--offload-arch=${RAD_GPU_TARGET}")
target_link_libraries(native_runtime_probe PRIVATE "${HIP_RUNTIME}")
enable_testing()
add_test(NAME native_runtime_probe COMMAND native_runtime_probe)
set_tests_properties(native_runtime_probe PROPERTIES TIMEOUT 60)
add_library(module_fixture SHARED module_fixture.cpp)
set_target_properties(module_fixture PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugin fixture")
add_executable(module_test module_test.cpp "${RADIANCE_SOURCE}/core/platform/module.cpp")
target_include_directories(module_test PRIVATE "${RADIANCE_SOURCE}/core")
target_link_libraries(module_test PRIVATE ${CMAKE_DL_LIBS})
add_test(NAME native_module_test COMMAND module_test "$<TARGET_FILE:module_fixture>")
set_tests_properties(native_module_test PROPERTIES TIMEOUT 10)
find_package(Threads REQUIRED)
add_executable(file_test file_test.cpp "${RADIANCE_SOURCE}/core/platform/file.cpp")
target_include_directories(file_test PRIVATE "${RADIANCE_SOURCE}/core")
target_link_libraries(file_test PRIVATE Threads::Threads)
add_test(NAME native_file_test COMMAND file_test "${CMAKE_BINARY_DIR}/file-test-owned-fixture.bin")
set_tests_properties(native_file_test PROPERTIES TIMEOUT 30)
add_executable(tensor_file_test tensor_file_test.cpp
  "${RADIANCE_SOURCE}/core/format/safetensors.cpp"
  "${RADIANCE_SOURCE}/core/format/imatrix.cpp"
  "${RADIANCE_SOURCE}/core/text/gguf.cpp"
  "${RADIANCE_SOURCE}/core/platform/file.cpp"
  "${RADIANCE_SOURCE}/core/util/rad_util.cpp")
target_include_directories(tensor_file_test PRIVATE "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi")
target_compile_definitions(tensor_file_test PRIVATE _CRT_SECURE_NO_WARNINGS)
add_test(NAME native_tensor_file_test COMMAND tensor_file_test "${CMAKE_BINARY_DIR}/tensor-test-owned-fixture.bin")
set_tests_properties(native_tensor_file_test PROPERTIES TIMEOUT 30)
add_executable(direct_file_test direct_file_test.cpp
  "${RADIANCE_SOURCE}/core/device/directio.cpp"
  "${RADIANCE_SOURCE}/core/platform/file.cpp"
  "${RADIANCE_SOURCE}/core/util/rad_util.cpp")
target_include_directories(direct_file_test PRIVATE "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi")
target_compile_definitions(direct_file_test PRIVATE _CRT_SECURE_NO_WARNINGS)
add_test(NAME native_direct_file_test COMMAND direct_file_test "${CMAKE_BINARY_DIR}/direct-test-tail-fix-fixture.bin")
set_tests_properties(native_direct_file_test PROPERTIES TIMEOUT 30)
