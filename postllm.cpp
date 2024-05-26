//#include "utils/builtins.h"
//#include "mb/pg_wchar.h"
//#include "fmgr.h"
//#include "common.h"
#include "postllm.hpp"
//#include "llama.cpp/llama.h"
//#include "llama.cpp/common/common.h"
#include "llama.h"
#include "common.h"
#include <unistd.h>
#include <fcntl.h>
#include <cmath>
#include <string>
#include <vector>

// TODO: common
// void llama_batch_clear(struct llama_batch & batch) {
//     batch.n_tokens = 0;
// }
// 
// void llama_batch_add(
//                  struct llama_batch & batch,
//                         llama_token   id,
//                           llama_pos   pos,
//     const std::vector<llama_seq_id> & seq_ids,
//                                bool   logits) {
//     batch.token   [batch.n_tokens] = id;
//     batch.pos     [batch.n_tokens] = pos;
//     batch.n_seq_id[batch.n_tokens] = seq_ids.size();
//     for (size_t i = 0; i < seq_ids.size(); ++i) {
//         batch.seq_id[batch.n_tokens][i] = seq_ids[i];
//     }
//     batch.logits  [batch.n_tokens] = logits;
// 
//     batch.n_tokens++;
// }


extern "C" {
    PG_MODULE_MAGIC;

    PGDLLEXPORT Datum prompt(PG_FUNCTION_ARGS);
    PG_FUNCTION_INFO_V1(prompt);
}

Datum prompt(PG_FUNCTION_ARGS) {
    text *model_filename_text = PG_GETARG_TEXT_P(0);
    text *prompt_text = PG_GETARG_TEXT_P(1);
    int32 n_len = PG_GETARG_INT32(2);

    char* model_filename = text_to_cstring(model_filename_text);
    char* prompt = text_to_cstring(prompt_text);
    // const int n_len = *total_tokens;

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_filename, model_params);

    if (model == NULL) {
        ereport(ERROR, (errcode_for_file_access(),
            errmsg("could not open file \"%s\"", model_filename)));
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.seed = 1234;
    ctx_params.n_ctx = n_len;
    ctx_params.n_threads = get_math_cpu_count();
    // ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;
    ctx_params.n_threads_batch = ctx_params.n_threads;
    // DEFAULT:
    // ctx_params.type_k = GGML_TYPE_F16;
    // COMPAT:
    // ctx_params.type_k = GGML_TYPE_Q4_K;

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, prompt, true);

    llama_batch batch = llama_batch_init(tokens_list.size(), 0, 1);
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    if (llama_decode(ctx, batch) != 0) {
        ereport(ERROR, (errcode_for_file_access(),
            errmsg(
            "%s: llama_decode() failed\n", __func__
        )));
    }

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    std::string res;

    while (n_cur <= n_len) {
        {
            elog(DEBUG1, "CALL llama_n_vocab");
            auto   n_vocab = llama_n_vocab(model);
            elog(DEBUG1, "CALL llama_get_logits_ith");
            auto * logits  = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            elog(DEBUG1, "CALL llama_sample_token_greedy");
            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            elog(DEBUG1, "CALL llama_token_is_eog");
            if (llama_token_is_eog(model, new_token_id) || n_cur == n_len) {
                res += "\n";

                break;
            }

            elog(DEBUG1, "CALL llama_token_to_piece");
            elog(DEBUG1, "RESULT: %s", llama_token_to_piece(ctx, new_token_id).c_str());
            // res = strcat(res, llama_token_to_piece(ctx->model, new_token_id).c_str());
            res += llama_token_to_piece(ctx, new_token_id);

            elog(DEBUG1, "CALL llama_batch_clear");
            llama_batch_clear(batch);

            elog(DEBUG1, "CALL llama_batch_add");
            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

            n_decode += 1;
        }

        n_cur += 1;

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        // ereport(ERROR, (errcode_for_file_access(),
        //     errmsg(
        //         "BREAKPOINT: first loop"
        // )));
    }

    text *res_text = cstring_to_text((char *) res.c_str());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    PG_RETURN_TEXT_P(res_text);
    // PG_RETURN_TEXT_P(prompt_text);
}

// }