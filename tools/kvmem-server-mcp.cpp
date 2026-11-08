#include "kvmem-server-mcp.h"
#include "httplib.h"
#include "server-mcp.h"

#include <csignal>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
void reply(httplib::Response & res, int status, const json & body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

// Browsers may reach a local listener even without an API key. Reject foreign
// origins and unrecognised Host names before exposing or invoking local tools.
bool allowed(const httplib::Request & req, httplib::Response & res, int port, bool authenticated) {
    const auto host = req.get_header_value("Host");
    const auto origin = req.get_header_value("Origin");
    const auto suffix = ":" + std::to_string(port);
    const bool local = host == "localhost" + suffix || host == "127.0.0.1" + suffix || host == "[::1]" + suffix ||
        (port == 80 && (host == "localhost" || host == "127.0.0.1" || host == "[::1]"));
    if ((!authenticated && !local) || (!origin.empty() && origin != "http://" + host && origin != "https://" + host)) {
        reply(res, 403, {{"error", "MCP tools require a same-origin request"}});
        return false;
    }
    return true;
}
}

void kvmem_install_mcp(httplib::Server & server, const std::string & config, int port, bool authenticated) {
    std::ifstream input(std::filesystem::u8path(config), std::ios::binary);
    if (!input) throw std::runtime_error("cannot open --mcp-servers-config file");
    std::ostringstream contents;
    contents << input.rdbuf();
    auto doc = json::parse_no_throw(contents.str());
    if (!doc.contains("mcpServers") || !doc.at("mcpServers").is_object())
        throw std::runtime_error("MCP configuration requires an mcpServers object");
    json enabled = json::object();
    for (const auto & [name, entry] : doc.at("mcpServers").items()) {
        if (entry.is_object() && entry.value("disabled", false)) continue;
        if (!entry.is_object() || entry.contains("url") || entry.value("command", std::string()).empty())
            throw std::runtime_error("MCP server '" + name + "' requires a stdio command");
        enabled[name] = entry;
    }
#ifdef SIGPIPE
    // A stdio child may exit between the liveness check and the pipe write.
    std::signal(SIGPIPE, SIG_IGN);
#endif
    auto manager = std::make_shared<server_mcp>();
    common_params params;
    params.mcp_servers_json = json({{"mcpServers", enabled}}).dump();
    manager->start(params);
    std::map<std::string, server_mcp_tool_def> tools;
    json listing = json::array();
    for (const auto & def : manager->list_tools()) {
        const auto name = def.server_name + "_" + def.name;
        if (!tools.emplace(name, def).second) throw std::runtime_error("duplicate MCP tool name: " + name);
        listing.push_back({{"tool", name}, {"display_name", name}, {"type", "mcp"},
            {"permissions", {{"write", false}}}, {"uses_cwd", false},
            {"definition", {{"type", "function"}, {"function", {{"name", name}, {"description", def.description},
                {"parameters", def.input_schema.is_object() ? def.input_schema : json::object()}}}}}});
    }
    server.Get("/tools", [listing = std::move(listing), port, authenticated](const auto & req, auto & res) {
        if (allowed(req, res, port, authenticated)) reply(res, 200, listing);
    });
    server.Post("/tools", [manager, tools = std::move(tools), port, authenticated](
            const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, port, authenticated)) return;
        const auto type = req.get_header_value("Content-Type");
        if (type.substr(0, type.find(';')) != "application/json")
            return reply(res, 415, {{"error", "expected application/json"}});
        auto body = json::parse_no_throw(req.body);
        if (!body.is_object() || !body.contains("tool") || !body.at("tool").is_string() ||
            (body.contains("params") && !body.at("params").is_object()))
            return reply(res, 400, {{"error", "expected tool name and object params"}});
        const auto it = tools.find(body.at("tool").get<std::string>());
        if (it == tools.end()) return reply(res, 404, {{"error", "unknown MCP tool"}});
        auto result = manager->call_tool(it->second.server_name, it->second.name,
                                         body.value("params", json::object()), req.is_connection_closed);
        reply(res, result.contains("error") ? 502 : 200, result);
    });
}
