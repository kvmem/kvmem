#include "llama.h"
#include "llama-kvmem-hooks.h"
#include "llama-kvmem-snapshot-stats.h"
#include "ggml-backend.h"
#include "kvmem/kv_snapshot.hpp"
#include <set>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static std::vector<uint8_t> state(llama_context * ctx, llama_state_seq_flags flags = 0) {
    std::vector<uint8_t> bytes(llama_state_seq_get_size_ext(ctx, 0, flags));
    require(llama_state_seq_get_data_ext(ctx, bytes.data(), bytes.size(), 0, flags) == bytes.size(),
            "cannot read native KV/recurrent state");
    return bytes;
}

static void check(llama_model * model, int rows) {
    llama_kvmem_params memory{};
    memory.enabled = true;
    memory.method = 1;
    memory.block_tokens = 32;
    memory.budget = 256;
    memory.gen_reserve = 128;
    memory.payload_host_bytes = 32u << 20;
    memory.query_begin = memory.query_end = memory.force_pos = -1;
    llama_kvmem_set_params(&memory);
    auto params = llama_context_default_params();
    params.n_ctx = 1024;
    params.n_batch = params.n_ubatch = 128;
    params.n_seq_max = 1;
    params.type_k = params.type_v = GGML_TYPE_Q8_0;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, params), llama_free);
    require(bool(ctx), "context initialization failed");
    std::vector<llama_token> tokens(rows + 32);
    for (int i = 0; i < (int) tokens.size(); ++i) tokens[i] = 100 + i;
    require(llama_decode(ctx.get(), llama_batch_get_one(tokens.data(), rows)) == 0, "prefill failed");
    llama_synchronize(ctx.get());
    const auto before = state(ctx.get());
    const auto checkpoint = state(ctx.get(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    using Bundle = std::unique_ptr<llama_kvmem_store_bundle, decltype(&llama_kvmem_store_bundle_free)>;
    Bundle parent(llama_kvmem_store_bundle_fork(), llama_kvmem_store_bundle_free);
    require(bool(parent), "cannot preserve parent attention snapshot");
    const auto immutable = llama_kvmem_checkpoint_capture(rows);
    require(bool(immutable), "immutable attention capture missing");
    const auto * image = llama_kvmem_checkpoint_image(*immutable);
    require(image && image->ready(), "immutable image is incomplete");
    const auto immutable_bytes = image->read();
    const auto repeated_image = llama_kvmem_checkpoint_capture(rows);
    const auto * repeated = llama_kvmem_checkpoint_image(*repeated_image);
    std::set<const void *> origins;
    image->account([&](const void * id, size_t) { origins.insert(id); });
    size_t shared = 0;
    repeated->account([&](const void * id, size_t bytes) { if (origins.count(id)) shared += bytes; });
    const auto blocks = (rows + 31) / 32;
    const auto expected_shared = image->size() / blocks * (rows / 32);
    require(shared == expected_shared, "full blocks did not share, or partial tail was aliased");
    if (rows == 65) {
        bool rejected = false;
        {
            std::vector<std::shared_ptr<const llama_kvmem_checkpoint_snapshot>> pressure;
            try {
                for (int i = 0; i < 128; ++i) pressure.push_back(llama_kvmem_checkpoint_capture(rows));
            } catch (const kvmem::MemoryError & error) {
                require(error.code() == kvmem::MemoryErrorCode::BudgetExceeded, "unexpected capture failure");
                rejected = true;
            }
        }
        require(rejected, "private-tail pressure never reached the shared H limit");
        require(image->read() == immutable_bytes, "failed capture damaged old source");
        require(llama_kvmem_checkpoint_restore(*immutable), "cannot restore source after H rejection");
        require(state(ctx.get()) == before, "H rejection changed native source state");
    }
    const auto captured = llama_kvmem_get_snapshot_stats();
    {
        Bundle unchanged(llama_kvmem_store_bundle_fork(), llama_kvmem_store_bundle_free);
        const auto repeated = llama_kvmem_get_snapshot_stats();
        require(repeated.packed_k_bytes == captured.packed_k_bytes &&
                repeated.packed_v_bytes == captured.packed_v_bytes,
                "unchanged fork recaptured existing packed KV");
    }
    std::printf("CHECKPOINT_CAPTURE rows=%d full_state_bytes=%zu partial_state_bytes=%zu host_capacity_bytes=%llu\n",
            rows, before.size(), checkpoint.size(),
            static_cast<unsigned long long>(llama_kvmem_store_capacity(rows)));
    std::vector<llama_pos> positions(32);
    for (int i = 0; i < 32; ++i) positions[i] = rows + i;
    auto extra = llama_batch_get_one(tokens.data() + rows, 32);
    extra.pos = extra.logical_pos = positions.data();
    require(llama_decode(ctx.get(), extra) == 0, "extra suffix prefill failed");
    llama_synchronize(ctx.get());
    auto selection = llama_kvmem_preview_retrieval();
    require(selection.blocks.size() == (tokens.size() + 31) / 32, "fixture block coverage is incomplete");
    selection.blocks = {0, 2};
    llama_kvmem_apply_selection(selection);
    require(llama_kvmem_get_attention_view().valid && llama_kvmem_get_attention_view().blocks.size() == 2,
            "sparse resident layout was not published");
    selection = llama_kvmem_preview_retrieval();
    selection.blocks = {0};
    llama_kvmem_apply_selection(selection);
    require(llama_kvmem_get_attention_view().blocks.size() == 1, "checkpoint suffix was not evicted");
    require(llama_state_seq_set_data_ext(ctx.get(), checkpoint.data(), checkpoint.size(), 0,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == checkpoint.size(), "recurrent checkpoint restore failed");
    require(llama_kvmem_remove_logical(ctx.get(), rows, -1), "suffix removal failed");
    llama_kvmem_truncate_cached(rows);
    require(llama_kvmem_restore_checkpoint_tail(), "cannot restore cold partial tail");
    // Normalize physical slot order before comparing the serialized bytes.
    llama_kvmem_apply_selection(llama_kvmem_preview_retrieval());
    require(llama_kvmem_get_attention_view().valid, "restored cells are incomplete");
    require(state(ctx.get()) == before, "restored native KV/recurrent bytes differ");

    const int gap = rows == 65 ? 0 : rows - 1;
    require(llama_kvmem_remove_logical(ctx.get(), gap, gap + 1), "cannot inject coverage gap");
    require(!llama_kvmem_restore_checkpoint_tail(), "resident coverage gap was hidden");
    require(!llama_kvmem_get_attention_view().valid, "broken view was published as reusable");
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    require(llama_decode(ctx.get(), llama_batch_get_one(tokens.data(), rows)) == 0, "cold recovery failed");
    llama_synchronize(ctx.get());
    require(state(ctx.get()) == before, "cold recovery retained invalid state");

    // Reuse native cells and logical block IDs with different accepted bytes.
    // The old parent's sealed records must survive both reset and this append.
    for (int i = rows; i < (int) tokens.size(); ++i) tokens[i] += 500;
    require(llama_decode(ctx.get(), extra) == 0, "branch suffix prefill failed");
    llama_synchronize(ctx.get());
    const auto branch_state = state(ctx.get());
    const auto branch_checkpoint = state(ctx.get(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    Bundle branch(llama_kvmem_store_bundle_fork(), llama_kvmem_store_bundle_free);
    require(image->read() == immutable_bytes, "old immutable snapshot changed after native ID reuse");
    require(llama_kvmem_checkpoint_restore(*immutable), "immutable parent attention restore failed");
    require(llama_state_seq_set_data_ext(ctx.get(), checkpoint.data(), checkpoint.size(), 0,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == checkpoint.size(), "immutable parent model restore failed");
    llama_kvmem_apply_selection(llama_kvmem_preview_retrieval());
    require(state(ctx.get()) == before, "immutable restored parent differs");
    require(llama_kvmem_checkpoint_restore(*immutable), "repeated immutable restore failed");
    require(image->read() == immutable_bytes, "restore mutated source snapshot");
    require(llama_kvmem_store_bundle_swap(parent.get()), "old parent attention restore failed");
    require(llama_state_seq_set_data_ext(ctx.get(), checkpoint.data(), checkpoint.size(), 0,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == checkpoint.size(), "old parent model-state restore failed");
    llama_kvmem_apply_selection(llama_kvmem_preview_retrieval());
    require(state(ctx.get()) == before, "old parent changed after native cell reuse");
    require(llama_kvmem_store_bundle_swap(branch.get()), "branch attention restore failed");
    require(llama_state_seq_set_data_ext(ctx.get(), branch_checkpoint.data(), branch_checkpoint.size(), 0,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == branch_checkpoint.size(), "branch model-state restore failed");
    llama_kvmem_apply_selection(llama_kvmem_preview_retrieval());
    require(state(ctx.get()) == branch_state, "branch changed after restoring old parent");
    llama_kvmem_store_bundle_reset(branch.get());
    parent.reset();
    require(state(ctx.get()) == branch_state, "releasing parked owners changed active branch");
    std::printf("PASS checkpoint tail %d/32: exact native bytes, coverage rejection, cold recovery\n", rows % 32);
    std::printf("PASS checkpoint branches rows=%d: native ID reuse, old parent restore, owner release\n", rows);
}

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) return argc == 1 ? 77 : 1;
    try {
        llama_backend_init();
        ggml_backend_load_all();
        auto params = llama_model_default_params();
        params.n_gpu_layers = argc == 3 ? std::stoi(argv[2]) : 99;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], params), llama_model_free);
        require(bool(model), "model loading failed");
        check(model.get(), 64);
        check(model.get(), 65);
        check(model.get(), 95);
        check(model.get(), 96);
        llama_kvmem_set_params(nullptr);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
