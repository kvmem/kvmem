// Small, model-free probe for the APIs required by Radiance's AQL backend.
// Compile against the chosen SDK; the runner should impose a process timeout.
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>

static int failures = 0;
static void library(const char * name, const void * symbol) {
    Dl_info info{};
    std::printf("library.%s=%s\n", name,
        dladdr(symbol, &info) && info.dli_fname ? info.dli_fname : "unknown");
}
static hsa_status_t agent(hsa_agent_t a, void *) {
    hsa_device_type_t type{};
    char name[64]{};
    hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &type);
    hsa_agent_get_info(a, HSA_AGENT_INFO_NAME, name);
    std::printf("agent=%s,type=%u\n", name, unsigned(type));
    if (type != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;
    uint32_t max_queue = 0;
    hsa_agent_get_info(a, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &max_queue);
    hsa_queue_t * queue = nullptr;
    const auto rc = hsa_queue_create(a, max_queue < 64 ? max_queue : 64,
        HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue);
    std::printf("aql.queue_create=%u\n", unsigned(rc));
    if (rc != HSA_STATUS_SUCCESS) ++failures;
    if (queue) hsa_queue_destroy(queue);
    return HSA_STATUS_SUCCESS;
}
int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("device.kfd=%d\ndevice.dxg=%d\n",
        access("/dev/kfd", R_OK | W_OK) == 0, access("/dev/dxg", R_OK | W_OK) == 0);
    library("hip", reinterpret_cast<const void *>(&hipGetDeviceCount));
    library("hsa", reinterpret_cast<const void *>(&hsa_init));
    int version = 0, count = 0;
    if (hipRuntimeGetVersion(&version) != hipSuccess) ++failures;
    auto hip_rc = hipGetDeviceCount(&count);
    std::printf("hip.version=%d\nhip.count=%d\nhip.status=%s\n", version, count, hipGetErrorName(hip_rc));
    if (hip_rc != hipSuccess || count == 0) ++failures;
    for (int i = 0; i < count; ++i) {
        char pci[64]{};
        const auto rc = hipDeviceGetPCIBusId(pci, sizeof pci, i);
        std::printf("hip.device.%d.pci=%s,status=%s\n", i, pci, hipGetErrorName(rc));
    }
    const auto rc = hsa_init();
    std::printf("hsa.init=%u\n", unsigned(rc));
    if (rc != HSA_STATUS_SUCCESS) return 2;
    if (hsa_iterate_agents(agent, nullptr) != HSA_STATUS_SUCCESS) ++failures;
    hsa_signal_t signal{};
    const auto sr = hsa_signal_create(0, 0, nullptr, &signal);
    std::printf("hsa.signal_create=%u\n", unsigned(sr));
    if (sr == HSA_STATUS_SUCCESS) hsa_signal_destroy(signal);
    else ++failures;
    hsa_shut_down();
    std::printf("probe.failures=%d\n", failures);
    return failures ? 1 : 0;
}
