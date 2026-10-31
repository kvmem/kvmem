set(RADIANCE_SOURCE_DIR "" CACHE PATH "Pinned Radiance source checkout")
if(NOT EXISTS "${RADIANCE_SOURCE_DIR}/core/mem/kv.h")
    message(FATAL_ERROR "Set RADIANCE_SOURCE_DIR to the pinned Radiance checkout (see docs/radiance-integration.md)")
endif()

# Only the adapter sees engine types. The portable kvmem library remains C++17
# and can still be built without Radiance, HIP, or any GPU.
add_library(kvmem_radiance_pages STATIC adapter/native_pages.cpp adapter/execution_pages.cpp)
target_compile_features(kvmem_radiance_pages PUBLIC cxx_std_20)
target_include_directories(kvmem_radiance_pages PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/adapter)
target_include_directories(kvmem_radiance_pages SYSTEM PRIVATE
    ${RADIANCE_SOURCE_DIR}/core ${RADIANCE_SOURCE_DIR}/abi)
target_link_libraries(kvmem_radiance_pages PUBLIC kvmem)
add_executable(radiance_rolling_pressure_test tests/rolling_pressure_test.cpp)
target_link_libraries(radiance_rolling_pressure_test PRIVATE kvmem_radiance_pages)
add_test(NAME radiance_rolling_pressure_test COMMAND radiance_rolling_pressure_test)
add_executable(radiance_native_pages_test tests/native_pages_test.cpp)
target_include_directories(radiance_native_pages_test SYSTEM PRIVATE
    ${RADIANCE_SOURCE_DIR}/core ${RADIANCE_SOURCE_DIR}/abi)
target_link_libraries(radiance_native_pages_test PRIVATE kvmem_radiance_pages)
add_test(NAME radiance_native_pages_test COMMAND radiance_native_pages_test)
add_executable(radiance_execution_pages_test tests/execution_pages_test.cpp)
target_link_libraries(radiance_execution_pages_test PRIVATE kvmem_radiance_pages)
add_test(NAME radiance_execution_pages_test COMMAND radiance_execution_pages_test)
add_executable(radiance_pool_geometry_test tests/pool_geometry_test.cpp)
target_link_libraries(radiance_pool_geometry_test PRIVATE kvmem_radiance_pages)
add_test(NAME radiance_pool_geometry_test COMMAND radiance_pool_geometry_test)
add_executable(radiance_query_identity_test tests/query_identity_test.cpp)
target_link_libraries(radiance_query_identity_test PRIVATE kvmem_radiance_pages)
add_test(NAME radiance_query_identity_test COMMAND radiance_query_identity_test)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME radiance_platform_launch_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/platform_launch_test.py)
    add_test(NAME radiance_adapter_pair_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/adapter_pair_test.py)
    add_test(NAME radiance_memory_accounting_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/memory_accounting_test.py)
    add_test(NAME radiance_linux_probe_metrics_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/linux_probe_metrics_test.py)
    add_test(NAME radiance_agent_history_fixture_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/agent_history_test.py)
    add_test(NAME radiance_selection_checks_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/selection_checks_test.py)
    add_test(NAME radiance_draft_control_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/draft_control_test.py)
    add_test(NAME radiance_target_compare_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/rad_target_compare_test.py)
    add_test(NAME radiance_mtp_quant_recipe_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/mtp_quant_recipe_test.py)
    add_test(NAME radiance_performance_pause_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/performance_pause_test.py)
endif()

# Device symbols are supplied by Radiance in the engine build, and by the fault
# injection fake in the portable tests. No HIP/OS types leak into the KVMem core.
add_library(kvmem_radiance_transfer STATIC adapter/page_transfer.cpp)
target_include_directories(kvmem_radiance_transfer SYSTEM PUBLIC ${RADIANCE_SOURCE_DIR}/abi)
target_link_libraries(kvmem_radiance_transfer PUBLIC kvmem_radiance_pages)
add_executable(radiance_page_transfer_test tests/page_transfer_test.cpp)
target_link_libraries(radiance_page_transfer_test PRIVATE kvmem_radiance_transfer)
add_test(NAME radiance_page_transfer_test COMMAND radiance_page_transfer_test)
if(TARGET rad_core)
    add_library(kvmem_radiance_memory STATIC adapter/native_memory.cpp adapter/statistics_capture.cpp adapter/statistics_observer.cpp)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/upstream.json" rad_upstream_json)
    string(JSON rad_upstream_revision GET "${rad_upstream_json}" revision)
    target_compile_definitions(kvmem_radiance_memory PRIVATE KVMEM_RADIANCE_UPSTREAM_REVISION="${rad_upstream_revision}")
    target_include_directories(kvmem_radiance_memory PRIVATE ${RADIANCE_SOURCE_DIR}/core)
    target_link_libraries(kvmem_radiance_memory PUBLIC kvmem_radiance_transfer rad_core)
    add_executable(kvmem-radiance-observer-probe tools/observer_main.cpp)
    target_include_directories(kvmem-radiance-observer-probe PRIVATE ${RADIANCE_SOURCE_DIR}/core)
    target_link_libraries(kvmem-radiance-observer-probe PRIVATE kvmem_radiance_memory)
    set_target_properties(kvmem-radiance-observer-probe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin ENABLE_EXPORTS ON)
    if(NOT WIN32)
        set_target_properties(kvmem-radiance-observer-probe PROPERTIES LINK_LIBRARY_OVERRIDE_rad_core WHOLE_ARCHIVE)
    endif()
    add_executable(radiance_native_memory_gpu_test tests/native_memory_gpu_test.cpp)
    target_include_directories(radiance_native_memory_gpu_test PRIVATE ${RADIANCE_SOURCE_DIR}/core)
    target_link_libraries(radiance_native_memory_gpu_test PRIVATE kvmem_radiance_memory)
    add_test(NAME radiance_native_memory_gpu_test COMMAND radiance_native_memory_gpu_test)
    set_tests_properties(radiance_native_memory_gpu_test PROPERTIES LABELS gpu TIMEOUT 60)
    add_executable(radiance_statistics_capture_gpu_test tests/statistics_capture_gpu_test.cpp)
    target_include_directories(radiance_statistics_capture_gpu_test PRIVATE ${RADIANCE_SOURCE_DIR}/core)
    target_link_libraries(radiance_statistics_capture_gpu_test PRIVATE kvmem_radiance_memory)
    add_test(NAME radiance_statistics_capture_gpu_test COMMAND radiance_statistics_capture_gpu_test)
    set_tests_properties(radiance_statistics_capture_gpu_test PROPERTIES LABELS gpu TIMEOUT 60)
    add_executable(radiance_page_transfer_gpu_test tests/page_transfer_gpu_test.cpp)
    target_include_directories(radiance_page_transfer_gpu_test PRIVATE ${RADIANCE_SOURCE_DIR}/core)
    target_link_libraries(radiance_page_transfer_gpu_test PRIVATE kvmem_radiance_transfer rad_core)
    add_test(NAME radiance_page_transfer_gpu_test COMMAND radiance_page_transfer_gpu_test)
    set_tests_properties(radiance_page_transfer_gpu_test PROPERTIES LABELS gpu TIMEOUT 60)
endif()
