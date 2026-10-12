#include "kvmem-responses.h"

// This TU deliberately includes only upstream server headers, so `json` here is
// common_json. See kvmem-responses.h for why the conversion crosses a string.
#include "server-chat.h"

#include <algorithm>
#include <stdexcept>
#include <string>

// A reasoning item a client sends back may carry its text only under `summary`.
//
// We emit every reasoning item with the text in both `summary` (type
// `summary_text`) and `content` (type `reasoning_text`), because OpenAI defines
// both and the two serve different readers. But @ai-sdk/openai keeps only the
// summary: when it replays the conversation it sends back
//
//     {"type": "reasoning", "summary": [{"type": "summary_text", "text": ...}]}
//
// with no `content` at all. server_chat_convert_responses_to_chatcmpl reads only
// `content[0].text` and rejects the item outright -- "item['content'] is not an
// array" -- which fails the whole request, tool loop included.
//
// Fold the summary into `content` before handing the body over, so upstream sees
// the shape it expects and the model keeps the reasoning it produced last turn
// instead of losing it. Items that already carry a `content` array are left
// untouched: that is the shape OpenAI's own clients send, and upstream handles it.
static void kvmem_responses_fold_reasoning_summary(json & body) {
    if (!body.contains("input") || !body.at("input").is_array()) {
        return;
    }
    for (json & item : body.at("input")) {
        if (!item.is_object() ||
            json_value(item, "type", std::string()) != "reasoning" ||
            item.contains("content")) {
            continue;
        }
        if (!item.contains("summary") || !item.at("summary").is_array()) {
            continue;
        }
        json content = json::array();
        for (const json & part : item.at("summary")) {
            if (part.is_object() && part.contains("text") && part.at("text").is_string()) {
                content.push_back(json{
                    {"text", part.at("text")},
                    {"type", "reasoning_text"},
                });
            }
        }
        if (content.empty()) {
            // Nothing worth keeping; give upstream the empty array its own check
            // wants so it reports the clearer "item['content'] is empty" instead
            // of failing on a missing key.
            content.push_back(json{{"text", ""}, {"type", "reasoning_text"}});
        }
        item["content"] = content;
    }
}

// A plain assistant message a client replays may arrive without a `type`.
//
// Upstream recognises an output message by `role == "assistant"` *and*
// `type == "message"`:
//
//     } else if (exists_and_is_string(item, "role") &&
//         item.at("role") == "assistant" &&
//         exists_and_is_string(item, "type") &&
//         item.at("type") == "message"
//
// @ai-sdk/openai replays the assistant turn as {"role": "assistant", "content":
// "..."} with no `type` at all, so that branch does not match, the item falls
// through every remaining branch, and the request dies on the final
// "Cannot determine type of 'item'". Supply the type it is looking for. The
// user/system/developer branch above matches on `role` alone, so those are left
// as they are.
static void kvmem_responses_fill_message_type(json & body) {
    if (!body.contains("input") || !body.at("input").is_array()) {
        return;
    }
    for (json & item : body.at("input")) {
        if (item.is_object() &&
            !item.contains("type") &&
            json_value(item, "role", std::string()) == "assistant") {
            item["type"] = "message";
        }
    }
}

// Merge after upstream validates and converts input_text parts to text.
// Qwen templates require a single leading system message.
static void kvmem_responses_merge_system_turns(json & body) {
    const auto is_system_turn = [](const json & item) {
        const std::string role = json_value(item, "role", std::string());
        return role == "system" || role == "developer";
    };
    const auto & messages = body.at("messages");
    if (std::none_of(messages.begin(), messages.end(), is_system_turn)) {
        return;
    }

    std::string merged;
    json kept = json::array({json{{"role", "system"}, {"content", ""}}});
    for (const json & message : messages) {
        if (!is_system_turn(message)) {
            kept.push_back(message);
            continue;
        }
        const auto & content = message.at("content");
        std::string text;
        if (content.is_string()) {
            text = content.get<std::string>();
        } else {
            for (const json & part : content) {
                if (json_value(part, "type", std::string()) != "text") {
                    throw std::invalid_argument("system message with non-text content is not supported");
                }
                text += part.at("text").get<std::string>();
            }
        }
        if (!text.empty()) {
            merged += (merged.empty() ? "" : "\n\n") + text;
        }
    }
    kept[0]["content"] = merged;
    body["messages"] = std::move(kept);
}

std::string kvmem_responses_to_chatcmpl(const std::string & body) {
    json parsed = json::parse(body);
    kvmem_responses_fold_reasoning_summary(parsed);
    kvmem_responses_fill_message_type(parsed);
    json converted = server_chat_convert_responses_to_chatcmpl(parsed);
    kvmem_responses_merge_system_turns(converted);
    return converted.dump();
}
