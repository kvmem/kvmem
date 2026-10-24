add_library(kvmem
    src/host/kvmem_store.cpp
    src/host/kvmem_runtime.cpp
    src/host/raw_kv_store.cpp
    src/host/mean_key_retrieval.cpp
    src/host/graph_recall.cpp
    src/host/memory_contract.cpp
    src/host/kv_content.cpp
    src/host/native_kv_payload.cpp
    src/host/host_kv_storage.cpp
    src/host/spill_io.cpp
    src/host/session_memory.cpp
    src/host/prefix_checkpoint.cpp
    src/host/rope.cpp
)
set_target_properties(kvmem PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(KVMEM_CORE_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "KVMem source root")

# Call only after the embedding backend has enabled CUDA. Host-only builds keep
# the independent reference and never acquire a CUDA runtime dependency.
# Optional CUDART_TARGET keeps the recall object and its consumers on the same
# runtime as the embedding backend (NInfer uses the static runtime on Windows).
function(kvmem_enable_cuda_recall)
    cmake_parse_arguments(KVMEM_RECALL "" "CUDART_TARGET" "" ${ARGN})
    if (NOT KVMEM_RECALL_CUDART_TARGET)
        set(KVMEM_RECALL_CUDART_TARGET CUDA::cudart)
    endif()
    # The standalone tests and embedding backend can both request this target.
    if (TARGET kvmem-recall-cuda)
        return()
    endif()
    if (NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "Enable CUDA before configuring its recall scorer")
    endif()
    find_package(CUDAToolkit REQUIRED)
    if (NOT TARGET ${KVMEM_RECALL_CUDART_TARGET})
        message(FATAL_ERROR "Unknown CUDA runtime target: ${KVMEM_RECALL_CUDART_TARGET}")
    endif()
    set(_kvmem_cuda_runtime Shared)
    if (KVMEM_RECALL_CUDART_TARGET STREQUAL "CUDA::cudart_static")
        set(_kvmem_cuda_runtime Static)
    endif()
    # The host target may have been created before CUDA was enabled in its
    # embedding parent. Compile CUDA in a newly created, enabled directory scope.
    add_library(kvmem-recall-cuda OBJECT "${KVMEM_CORE_SOURCE_ROOT}/src/cuda/page_softmax.cu")
    target_include_directories(kvmem-recall-cuda PRIVATE "${KVMEM_CORE_SOURCE_ROOT}/include")
    set_target_properties(kvmem-recall-cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON CUDA_RUNTIME_LIBRARY ${_kvmem_cuda_runtime})
    target_sources(kvmem PRIVATE $<TARGET_OBJECTS:kvmem-recall-cuda>)
    target_link_libraries(kvmem PUBLIC ${KVMEM_RECALL_CUDART_TARGET})
    target_compile_definitions(kvmem PUBLIC KVMEM_PAGE_SOFTMAX_CUDA=1)
    if (MSVC)
        target_compile_options(kvmem-recall-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endfunction()
target_include_directories(kvmem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kvmem PUBLIC cxx_std_17)
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # ninfer's /d1trimfile prefix is its own source root and does not match this directory.
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/d1trimfile:${CMAKE_CURRENT_SOURCE_DIR}/>)
endif()

if (WIN32)
    # std::getenv is standard C++ and portable; MSVC marks it deprecated in
    # favour of the Windows-only _dupenv_s. Make the same choice llama.cpp makes
    # for its own sources rather than edit portable code.
    target_compile_definitions(kvmem PRIVATE _CRT_SECURE_NO_WARNINGS)
    # On ELF a shared library exports every symbol, which is what
    # `target_link_libraries(llama PRIVATE kvmem)` relies on. PE has no such
    # default, so without this libllama fails to link with undefined
    # kvmem::* symbols. Let CMake generate the export list.
    set_target_properties(kvmem PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

function(kvmem_add_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE kvmem)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

kvmem_add_test(backend_rebind_test)
kvmem_add_test(kvmem_store_test)
kvmem_add_test(pinned_kv_tier_test)
kvmem_add_test(memory_contract_test)
kvmem_add_test(memory_elastic_decode_test)
kvmem_add_test(native_kv_payload_test)
kvmem_add_test(host_kv_storage_test)
kvmem_add_test(spill_io_test)
kvmem_add_test(tiered_kv_storage_test)
kvmem_add_test(native_residency_test)
kvmem_add_test(kv_content_test)
kvmem_add_test(prefix_checkpoint_test)
kvmem_add_test(mean_key_retrieval_test)
kvmem_add_test(graph_recall_test)
kvmem_add_test(query_policy_test)
kvmem_add_test(kvmem_runtime_test)
kvmem_add_test(raw_kv_store_test)
