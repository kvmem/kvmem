#pragma once

#include "httplib.h"
#include <stdexcept>

// Inference executes on HTTP workers. A hardware-sized or expanding pool can
// spread successive host-KV allocations over many allocator arenas, retaining
// a document-sized high-water mark in each. Keep a fixed pool with at least
// four workers so a single lane can serve monitoring during generation.
inline int kvmem_configure_http_pool(httplib::Server & server, int parallel, int requested = -1) {
    if (parallel < 1 || parallel > 8) throw std::invalid_argument("HTTP pool requires 1..8 inference lanes");
    const int automatic = parallel == 1 ? 4 : 2 * parallel;
    const int workers = requested > 0 ? requested : automatic;
    server.new_task_queue = [workers] {
        return new httplib::ThreadPool(static_cast<size_t>(workers), static_cast<size_t>(workers));
    };
    return workers;
}
