#pragma once

#include <string>

namespace httplib { class Server; }

void kvmem_install_mcp(httplib::Server & server, const std::string & config, int port, bool authenticated);
