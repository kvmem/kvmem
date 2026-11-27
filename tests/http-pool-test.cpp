#ifdef NDEBUG
#undef NDEBUG
#endif
#include "kvmem-http-pool.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int main() {
    for (int lanes : {1, 2, 4, 8}) {
        httplib::Server server;
        const int expected = lanes == 1 ? 4 : 2 * lanes;
        assert(kvmem_configure_http_pool(server, lanes) == expected);
        assert(kvmem_configure_http_pool(server, lanes, 0) == expected);
        assert(kvmem_configure_http_pool(server, lanes, -3) == expected);
        assert(kvmem_configure_http_pool(server, lanes, 19) == 19);
    }
    for (int lanes : {0, 9}) {
        httplib::Server server;
        bool rejected = false;
        try { kvmem_configure_http_pool(server, lanes); } catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    {
        httplib::Server explicit_single;
        assert(kvmem_configure_http_pool(explicit_single, 1, 1) == 1);
    }

    httplib::Server server;
    constexpr int worker_count = 4;
    assert(kvmem_configure_http_pool(server, 1) == worker_count);
    std::mutex mutex;
    std::condition_variable changed;
    std::set<std::thread::id> threads;
    int entered = 0;
    bool release = false;
    auto remember = [&] {
        std::lock_guard<std::mutex> lock(mutex);
        threads.insert(std::this_thread::get_id());
    };
    server.Get("/health", [&](const httplib::Request &, httplib::Response & response) {
        remember(); response.set_content("OK", "text/plain");
    });
    server.Get("/hold", [&](const httplib::Request &, httplib::Response & response) {
        remember();
        std::unique_lock<std::mutex> lock(mutex);
        ++entered; changed.notify_all();
        changed.wait(lock, [&] { return release; });
        response.set_content("done", "text/plain");
    });
    server.Get("/stream", [&](const httplib::Request &, httplib::Response & response) {
        response.set_chunked_content_provider("text/event-stream", [&](size_t, httplib::DataSink & sink) {
            remember();
            sink.write("data: done\n\n", 12); sink.done(); return true;
        });
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    assert(port > 0);
    std::thread serving([&] { assert(server.listen_after_bind()); });
    server.wait_until_ready();
    auto get = [&](const char * path) {
        httplib::Client client("127.0.0.1", port);
        client.set_read_timeout(5, 0);
        auto response = client.Get(path);
        assert(response && response->status == 200);
        return response->body;
    };
    auto first = std::async(std::launch::async, [&] { return get("/hold"); });
    {
        std::unique_lock<std::mutex> lock(mutex);
        assert(changed.wait_for(lock, 3s, [&] { return entered == 1; }));
    }
    auto second = std::async(std::launch::async, [&] { return get("/hold"); });
    {
        std::unique_lock<std::mutex> lock(mutex);
        assert(changed.wait_for(lock, 3s, [&] { return entered == 2; }));
    }
    // Generation and a second occupied HTTP connection must leave room for monitoring.
    auto health = std::async(std::launch::async, [&] { return get("/health"); });
    assert(health.wait_for(3s) == std::future_status::ready && health.get() == "OK");
    std::vector<std::future<std::string>> queued;
    for (int i = 2; i < worker_count; ++i) queued.push_back(std::async(std::launch::async, [&] { return get("/hold"); }));
    {
        std::unique_lock<std::mutex> lock(mutex);
        assert(changed.wait_for(lock, 3s, [&] { return entered == worker_count; }));
    }
    for (int i = 0; i < 8; ++i) queued.push_back(std::async(std::launch::async, [&] { return get("/hold"); }));
    {
        std::unique_lock<std::mutex> lock(mutex);
        // Overflow requests stay queued rather than creating new allocator arenas.
        assert(!changed.wait_for(lock, 200ms, [&] { return entered > worker_count; }));
        release = true;
    }
    changed.notify_all();
    assert(first.get() == "done" && second.get() == "done");
    for (auto & response : queued) assert(response.get() == "done");
    for (int i = 0; i < 24; ++i) assert(get("/health") == "OK");
    assert(get("/stream") == "data: done\n\n");
    server.stop(); serving.join();
    assert(entered == worker_count + 8 && threads.size() <= worker_count);
    std::cout << "HTTP pool: fixed worker bound, queued overflow, health and streaming passed\n";
}
