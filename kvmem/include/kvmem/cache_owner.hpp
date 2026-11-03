#pragma once
#include <optional>
#include <stdexcept>
#include <string>

namespace kvmem {
// Owners identify mutable conversation lineages. They do not authorize reuse;
// native token/media/position, accepted-state and Query proofs remain required.
struct CacheOwnerHints {
    std::optional<std::string> conversation, prompt, sdk_prompt, legacy_session;
};
inline std::optional<std::string> resolve_cache_owner(const CacheOwnerHints& hints) {
    const auto valid=[](const std::optional<std::string>& key,bool empty) {
        if(key && (key->size()>256 || (!empty && key->empty())))
            throw std::invalid_argument("cache owner must contain " + std::string(empty?"0":"1") + " to 256 bytes");
    };
    valid(hints.conversation,false);valid(hints.prompt,true);valid(hints.sdk_prompt,true);
    auto chosen=hints.prompt;
    if(chosen && chosen->empty())chosen.reset();
    if(hints.sdk_prompt && !hints.sdk_prompt->empty()) {
        if(chosen && *chosen!=*hints.sdk_prompt)
            throw std::invalid_argument("prompt_cache_key and promptCacheKey must agree");
        chosen=hints.sdk_prompt;
    }
    if(hints.conversation)return hints.conversation;
    if(chosen)return chosen;
    // session_id was a display-only external extension. Preserve acceptance
    // of long labels, without turning a truncated label into a cache owner.
    if(hints.legacy_session && !hints.legacy_session->empty() && hints.legacy_session->size()<=256)return hints.legacy_session;
    return std::nullopt;
}
// Discovery of a different lineage is a new-user Query operation. Restoring a
// proven continuation is separate, and never grants cold-history reselection.
inline bool may_discover_shared_prefix(bool current_user_query) { return current_user_query; }
} // namespace kvmem
