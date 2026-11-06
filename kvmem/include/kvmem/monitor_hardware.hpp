#pragma once
#include "monitor.hpp"
#include <condition_variable>
#include <fstream>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#if __has_include(<nvml.h>)
// nvml.h defines a one-argument DEPRECATED macro that collides with llama.h's
// two-argument DEPRECATED used by mtmd headers. Keep the scope local.
#pragma push_macro("DEPRECATED")
#include <nvml.h>
#pragma pop_macro("DEPRECATED")
#define KVMEM_MONITOR_NVML 1
#endif

namespace kvmem {
// A dedicated sampler; HTTP readers never call a GPU driver or take an inference lock.
class monitor_hardware {
    using json = nlohmann::json;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
    json data_ = {{"sampled_at_ms", nullptr}, {"gpus", json::array()},
                  {"cpu_percent", nullptr}, {"ram_used_bytes", nullptr}, {"ram_total_bytes", nullptr}};
    uint64_t previous_total_ = 0, previous_idle_ = 0;
#ifdef _WIN32
    HMODULE library_ = nullptr;
    void * symbol(const char * name) { return reinterpret_cast<void *>(GetProcAddress(library_, name)); }
#else
    void * library_ = nullptr;
    void * symbol(const char * name) { return dlsym(library_, name); }
#endif
#ifdef KVMEM_MONITOR_NVML
    bool nvml_ready_ = false;
    template<class F> F get(const char * name) { return reinterpret_cast<F>(symbol(name)); }
    void gpu_sample(json & result) {
        if (!nvml_ready_) return;
        const auto count_fn = get<decltype(&nvmlDeviceGetCount_v2)>("nvmlDeviceGetCount_v2");
        const auto handle_fn = get<decltype(&nvmlDeviceGetHandleByIndex_v2)>("nvmlDeviceGetHandleByIndex_v2");
        if (!count_fn || !handle_fn) return;
        unsigned count = 0;
        if (count_fn(&count) != NVML_SUCCESS) return;
        for (unsigned i = 0; i < count; ++i) {
            nvmlDevice_t device{};
            if (handle_fn(i, &device) != NVML_SUCCESS) continue;
            json gpu = {{"uuid", nullptr}, {"name", nullptr}, {"pci_bus_id", nullptr}, {"utilization_percent", nullptr},
                {"used_bytes", nullptr}, {"total_bytes", nullptr}, {"temperature_c", nullptr},
                {"power_w", nullptr}, {"pcie_generation", nullptr}, {"pcie_width", nullptr},
                {"pcie_rx_bytes_per_second", nullptr}};
            char name[128]{};
            if (auto fn = get<decltype(&nvmlDeviceGetUUID)>("nvmlDeviceGetUUID"))
                if (fn(device, name, sizeof(name)) == NVML_SUCCESS) gpu["uuid"] = name;
            if (auto fn = get<decltype(&nvmlDeviceGetName)>("nvmlDeviceGetName"))
                if (fn(device, name, sizeof(name)) == NVML_SUCCESS) gpu["name"] = name;
            nvmlPciInfo_t pci{};
            if (auto fn = get<decltype(&nvmlDeviceGetPciInfo_v3)>("nvmlDeviceGetPciInfo_v3"))
                if (fn(device, &pci) == NVML_SUCCESS) gpu["pci_bus_id"] = pci.busId;
            nvmlUtilization_t utilization{};
            if (auto fn = get<decltype(&nvmlDeviceGetUtilizationRates)>("nvmlDeviceGetUtilizationRates"))
                if (fn(device, &utilization) == NVML_SUCCESS) gpu["utilization_percent"] = utilization.gpu;
            nvmlMemory_t memory{};
            if (auto fn = get<decltype(&nvmlDeviceGetMemoryInfo)>("nvmlDeviceGetMemoryInfo"))
                if (fn(device, &memory) == NVML_SUCCESS) { gpu["used_bytes"] = memory.used; gpu["total_bytes"] = memory.total; }
            unsigned value = 0;
            if (auto fn = get<decltype(&nvmlDeviceGetTemperature)>("nvmlDeviceGetTemperature"))
                if (fn(device, NVML_TEMPERATURE_GPU, &value) == NVML_SUCCESS) gpu["temperature_c"] = value;
            if (auto fn = get<decltype(&nvmlDeviceGetPowerUsage)>("nvmlDeviceGetPowerUsage"))
                if (fn(device, &value) == NVML_SUCCESS) gpu["power_w"] = value / 1000.0;
            if (auto fn = get<decltype(&nvmlDeviceGetCurrPcieLinkGeneration)>("nvmlDeviceGetCurrPcieLinkGeneration"))
                if (fn(device, &value) == NVML_SUCCESS) gpu["pcie_generation"] = value;
            if (auto fn = get<decltype(&nvmlDeviceGetCurrPcieLinkWidth)>("nvmlDeviceGetCurrPcieLinkWidth"))
                if (fn(device, &value) == NVML_SUCCESS) gpu["pcie_width"] = value;
            if (auto fn = get<decltype(&nvmlDeviceGetPcieThroughput)>("nvmlDeviceGetPcieThroughput"))
                if (fn(device, NVML_PCIE_UTIL_RX_BYTES, &value) == NVML_SUCCESS) gpu["pcie_rx_bytes_per_second"] = uint64_t(value) * 1024;
            if (!gpu["uuid"].is_null()) result["gpus"].push_back(std::move(gpu));
        }
    }
#endif
    void sample() {
        json next = {{"sampled_at_ms", monitor::now_ms()}, {"gpus", json::array()},
            {"cpu_percent", nullptr}, {"ram_used_bytes", nullptr}, {"ram_total_bytes", nullptr}};
        uint64_t total = 0, idle = 0;
#ifdef _WIN32
        FILETIME idle_time{}, kernel{}, user{};
        const auto ticks = [](FILETIME t) { return (uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
        if (GetSystemTimes(&idle_time, &kernel, &user)) { total = ticks(kernel) + ticks(user); idle = ticks(idle_time); }
        MEMORYSTATUSEX memory{}; memory.dwLength = sizeof(memory);
        if (GlobalMemoryStatusEx(&memory)) {
            next["ram_total_bytes"] = memory.ullTotalPhys;
            next["ram_used_bytes"] = memory.ullTotalPhys - memory.ullAvailPhys;
        }
#elif defined(__linux__)
        std::ifstream cpu("/proc/stat");
        std::string label;
        uint64_t user = 0, nice = 0, system = 0, io = 0, irq = 0, soft = 0, steal = 0;
        if (cpu >> label >> user >> nice >> system >> idle >> io >> irq >> soft >> steal) {
            idle += io; total = user + nice + system + idle + irq + soft + steal;
        }
        std::ifstream memory("/proc/meminfo");
        std::string line; uint64_t bytes = 0, available = 0;
        while (std::getline(memory, line)) {
            if (line.rfind("MemTotal:", 0) == 0) bytes = std::stoull(line.substr(9)) * 1024;
            if (line.rfind("MemAvailable:", 0) == 0) available = std::stoull(line.substr(13)) * 1024;
        }
        if (bytes && available <= bytes) { next["ram_total_bytes"] = bytes; next["ram_used_bytes"] = bytes - available; }
#endif
        if (previous_total_ && total > previous_total_ && idle >= previous_idle_)
            next["cpu_percent"] = std::clamp(100.0 * (1.0 - double(idle - previous_idle_) / double(total - previous_total_)), 0.0, 100.0);
        previous_total_ = total; previous_idle_ = idle;
#ifdef KVMEM_MONITOR_NVML
        gpu_sample(next);
#endif
        std::lock_guard<std::mutex> lock(mutex_);
        data_ = std::move(next);
    }
    void run() {
#ifdef KVMEM_MONITOR_NVML
#ifdef _WIN32
        library_ = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
        library_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
#endif
        if (library_) if (auto init = get<decltype(&nvmlInit_v2)>("nvmlInit_v2")) nvml_ready_ = init() == NVML_SUCCESS;
#endif
        for (;;) {
            try { sample(); } catch (...) { /* A telemetry failure must not terminate inference. */ }
            std::unique_lock<std::mutex> lock(mutex_);
            if (wake_.wait_for(lock, std::chrono::seconds(1), [&] { return stopping_; })) break;
        }
#ifdef KVMEM_MONITOR_NVML
        if (nvml_ready_) if (auto shutdown = get<decltype(&nvmlShutdown)>("nvmlShutdown")) shutdown();
        if (library_) {
#ifdef _WIN32
            FreeLibrary(library_);
#else
            dlclose(library_);
#endif
        }
#endif
    }
    std::thread worker_;
public:
    monitor_hardware() : worker_([this] { run(); }) {}
    ~monitor_hardware() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        wake_.notify_one();
        worker_.join();
    }
    json snapshot() const { std::lock_guard<std::mutex> lock(mutex_); return data_; }
};
} // namespace kvmem
