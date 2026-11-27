#ifdef NDEBUG
#undef NDEBUG
#endif
#include "kvmem-http-pool.h"
#include "kvmem/raw_kv_store.hpp"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>
#ifdef __GLIBC__
#include <malloc.h>

static uint64_t rss_bytes() {
    std::ifstream stream("/proc/self/smaps_rollup");
    std::string line;
    while (std::getline(stream, line)) {
        if (line.rfind("Rss:", 0) == 0) return std::stoull(line.substr(4)) * 1024;
    }
    throw std::runtime_error("missing process RSS");
}

int main(int argc, char ** argv) {
    // The optional diagnostic control reproduces the original one-lane pool.
    const bool legacy = argc == 2 && std::string(argv[1]) == "--unbounded-control";
    const uint32_t tokens = 8192, block = 32, layers = 16, row_bytes = 544;
    const uint64_t payload = uint64_t(tokens) * layers * row_bytes * 2;
    auto storage = std::make_shared<kvmem::HostKvStorage>();
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = layers; cfg.n_embd_k = cfg.n_embd_v = 512; cfg.block_tokens = block;
    cfg.k_gpu_row_bytes = cfg.v_gpu_row_bytes = row_bytes; cfg.payload_storage = storage;
    std::unique_ptr<kvmem::RawKvStore> current;
    std::mutex store_mutex;
    httplib::Server server;
    constexpr int worker_count = 4;
    if (!legacy) assert(kvmem_configure_http_pool(server, 1) == worker_count);
    std::set<std::thread::id> workers;
    server.Post("/document", [&](const httplib::Request &, httplib::Response & response) {
        std::lock_guard<std::mutex> lock(store_mutex);
        // Requests are sequential; keep only one completed store, with a
        // transient old/new overlap matching a conversation handover.
        workers.insert(std::this_thread::get_id());
        auto old = std::move(current);
        current = std::make_unique<kvmem::RawKvStore>(cfg);
        std::vector<uint8_t> data(block * row_bytes, 17);
        std::vector<float> mean(512, 1.0f);
        for (uint32_t pos = 0; pos < tokens; pos += block) {
            for (uint32_t layer = 0; layer < layers; ++layer) {
                current->write_layer_k_gpu(pos, block, layer, data.data());
                current->write_layer_v_gpu(pos, block, layer, data.data());
                current->write_layer_mean_sum(pos, block, layer, mean.data());
            }
        }
        old.reset();
        assert(storage->charged_bytes() == payload);
        response.set_content("OK", "text/plain");
    });
    server.Post("/question", [&](const httplib::Request &, httplib::Response & response) {
        std::lock_guard<std::mutex> lock(store_mutex);
        workers.insert(std::this_thread::get_id());
        assert(storage->charged_bytes() == payload);
        response.set_content("OK", "text/plain");
    });
    server.Post("/clear", [&](const httplib::Request &, httplib::Response & response) {
        std::lock_guard<std::mutex> lock(store_mutex);
        current.reset(); assert(storage->charged_bytes() == 0);
        response.set_content("OK", "text/plain");
    });
    const int port = server.bind_to_any_port("127.0.0.1"); assert(port > 0);
    std::thread serving([&] { assert(server.listen_after_bind()); });
    server.wait_until_ready();
    auto post = [&](const char * path) {
        httplib::Client client("127.0.0.1", port); client.set_read_timeout(15, 0);
        auto response = client.Post(path, "", "text/plain");
        assert(response && response->status == 200 && response->body == "OK");
    };
    uint64_t warm = 0, last = 0;
    for (int i = 1; i <= 12; ++i) {
        post("/document"); post("/question");
        last = rss_bytes(); if (i == 4) warm = last;
        const auto heap = mallinfo2();
        std::cout << "case=" << i << " rss_bytes=" << last << " charged=" << storage->charged_bytes()
                  << " arena_free=" << heap.fordblks << '\n';
    }
    // Allow half a live payload of noise after warm-up, but reject the
    // document-sized-per-new-worker slope seen with the old default.
    const bool plateau = last <= warm + payload / 2;
    post("/clear");
    const auto freed = rss_bytes(); const auto heap = mallinfo2();
    malloc_trim(0); const auto trimmed = rss_bytes();
    server.stop(); serving.join();
    std::cout << "legacy=" << legacy << " workers=" << workers.size() << " warm=" << warm
              << " last=" << last << " freed=" << freed << " trimmed=" << trimmed
              << " live=" << heap.uordblks << " free=" << heap.fordblks
              << " charged_final=" << storage->charged_bytes() << " plateau=" << plateau << '\n';
    if (!legacy) assert(workers.size() <= worker_count);
    return plateau ? 0 : 1;
}
#else
int main() { std::cout << "glibc allocator regression requires Linux/glibc\n"; return 77; }
#endif
