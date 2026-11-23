#include "llama.h"
#include "llama-batch.h"
#include "llama-model.h"
#include "llama-kvmem-hooks.h"
#include "llama-memory-kvmem-iswa.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : std::getenv("KVMEM_TEST_GEMMA4_MODEL");
    if (!path || !*path) return 77;
    try {
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 99;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
                llama_model_load_from_file(path, model_params), llama_model_free);
        require(model && model->arch == LLM_ARCH_GEMMA4, "a Gemma-4 model is required");
        llama_kvmem_params memory{};
        memory.enabled = true;
        memory.budget = 4096;
        memory.gen_reserve = 512;
        memory.block_tokens = 64;
        memory.method = 1;
        memory.query_begin = memory.query_end = memory.force_pos = -1;
        llama_kvmem_set_params(&memory);
        auto params = llama_context_default_params();
        params.n_ctx = 8192;
        params.n_batch = 4096;
        params.n_ubatch = 128;
        params.n_seq_max = 1;
        params.type_k = params.type_v = GGML_TYPE_Q8_0;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
                llama_init_from_model(model.get(), params), llama_free);
        require(bool(context), "context creation failed");
        auto * iswa = dynamic_cast<llama_memory_kvmem_iswa *>(llama_get_memory(context.get()));
        require(iswa != nullptr, "Gemma-4 did not use the KVMem iSWA wrapper");
        const uint32_t count = iswa->get_swa()->get_size() + 1;
        require(count < memory.budget, "fixture must fit global KV but exceed the SWA cache");
        llama_batch_ext batch(count, model->hparams.n_embd, 0, 1, iswa,
                              llama_vocab_n_tokens(llama_model_get_vocab(model.get())), 1);
        for (uint32_t i = 0; i < count; ++i) {
            const int index = batch.add_token(0);
            const llama_pos pos = i;
            require(batch.set_token_id(index, 100) && batch.set_token_pos(index, &pos), "batch setup failed");
            batch.set_output(index, i + 1 == count);
        }
        llama_batch_allocr allocator(1);
        require(allocator.init(batch, model->vocab, false), "batch allocator rejected fixture");
        const auto before = iswa->attn_kvmem()->store_n_tokens();
        // Deliberately request one oversized ubatch through the memory API.
        // This isolates SWA rejection from the global pool's valid capacity.
        auto failed = iswa->init_batch(allocator, count, false);
        require(failed && failed->get_status() == LLAMA_MEMORY_STATUS_FAILED_PREPARE,
                "oversized SWA ubatch was not rejected");
        const auto after = iswa->attn_kvmem()->store_n_tokens();
        std::printf("SWA rejected %u rows; global rows before=%u after=%u\n", count, before, after);
        require(before == after, "failed SWA prepare mutated the global logical history");
        std::vector<llama_token> valid(16, 100);
        require(llama_decode(context.get(), llama_batch_get_one(valid.data(), valid.size())) == 0,
                "valid decode failed after SWA rejection");
        require(iswa->attn_kvmem()->store_n_tokens() == valid.size(), "recovery appended duplicate rows");
        std::puts("Gemma-4 prepare failure recovery passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
