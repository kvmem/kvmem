#include "platform/module.h"
#include <cstdio>
#include <string>
int main(int argc, char** argv) {
    if (argc != 2) return 1;
    std::string error;
    auto* h = rad::platform::module_open(argv[1], &error);
    if (!h) { std::fprintf(stderr,"%s\n",error.c_str()); return 2; }
    auto fn = reinterpret_cast<int(*)()>(rad::platform::module_symbol(h,"platform_fixture_value"));
    const bool ok = fn && fn() == 417 && !rad::platform::module_symbol(h,"not_exported");
    rad::platform::module_close(h);
    if (!ok) return 3;
    if (rad::platform::module_open(std::string(argv[1])+".missing", &error) || error.empty()) return 4;
    std::puts("PASS: native module load, exact handle symbol, missing symbol, unload, absent file");
    return 0;
}
