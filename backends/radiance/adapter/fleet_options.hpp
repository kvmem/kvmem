#pragma once
#include "fleet/config.h"
#include <cctype>
#include <filesystem>
#include <stdexcept>

namespace kvmem::radiance {
// The launcher still configures one model. Carry its already-validated Config
// directly, preserving repeated-option precedence and command-line provenance.
inline rad::fleet::FleetConfig single_model_fleet(rad::Config cfg, int argc, char** argv) {
    const auto valid_name=[](const std::string& name) {
        if(name.empty())return false;
        for(unsigned char ch:name)if(ch==':' || std::isspace(ch))return false;
        return true;
    };
    if(cfg.served_model_name.empty()) {
        auto path=std::filesystem::path(cfg.model).filename();
        if(path.extension()==".rad")path=path.stem();
        cfg.served_model_name=path.string();
        if(!valid_name(cfg.served_model_name))cfg.served_model_name="radiance-local";
    } else if(!valid_name(cfg.served_model_name)) {
        throw std::invalid_argument("Radiance v2 model names must not contain whitespace or ':'");
    }
    rad::fleet::FleetConfig fleet;
    fleet.path="KVMem command line";
    fleet.cards=cfg.devices;
    rad::fleet::ModelEntry entry;
    entry.name=cfg.served_model_name;
    entry.cfg=std::move(cfg);
    for(int i=1;i<argc;++i)entry.args.emplace_back(argv[i]);
    fleet.models.push_back(std::move(entry));
    return fleet;
}
}
