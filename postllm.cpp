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

    void _PG_init(void);
    void _PG_fini(void);

    PGDLLEXPORT Datum text_to_tokens(PG_FUNCTION_ARGS);
    PG_FUNCTION_INFO_V1(text_to_tokens);

    PGDLLEXPORT Datum text_to_token_length(PG_FUNCTION_ARGS);
    PG_FUNCTION_INFO_V1(text_to_token_length);

    PGDLLEXPORT Datum model_n_ctx(PG_FUNCTION_ARGS);
    PG_FUNCTION_INFO_V1(model_n_ctx);

    PGDLLEXPORT Datum load_model(PG_FUNCTION_ARGS);
    PG_FUNCTION_INFO_V1(load_model);

    PGDLLEXPORT Datum free_model(PG_FUNCTION_ARGS);
    PG_FUNCTION_INFO_V1(free_model);

    PGDLLEXPORT Datum prompt_model(PG_FUNCTION_ARGS);
    PG_FUNCTION_INFO_V1(prompt_model);
}

typedef struct
{
    char *model_name;
    llama_model *model;
} SharedModel;

static HTAB *shared_models = NULL;
static LWLock *ModelHashLock = NULL;
static char *lock_name = "model_hash_lock";

static void request_shmem(void) {
    elog(INFO, "Creating lock");
    RequestNamedLWLockTranche(lock_name, 1);
}

static void startup_shmem(void) {
    elog(INFO, "Assigning lock to global");
    ModelHashLock = &(GetNamedLWLockTranche(lock_name))->lock;

    elog(INFO, "Initializing model hash table");
    HASHCTL ctl;
    memset(&ctl, 0, sizeof(ctl));
    ctl.keysize = sizeof(char *);
    ctl.entrysize = sizeof(SharedModel);
    shared_models = ShmemInitHash("Shared Models Hash",
                                  16,  /* Initial size */
                                  128, /* Maximum size */
                                  &ctl,
                                  HASH_ELEM | HASH_BLOBS);
}

void _PG_init(void)
{
    if (!process_shared_preload_libraries_in_progress)
        ereport(ERROR, (errmsg("postllm must be preloaded on database startup")));

    shmem_request_hook = request_shmem;
    shmem_startup_hook = startup_shmem;

    elog(INFO, "Initializing llama backend");
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
}

void _PG_fini(void)
{
    HASH_SEQ_STATUS status;
    SharedModel *entry;

    hash_seq_init(&status, shared_models);
    while ((entry = (SharedModel *) hash_seq_search(&status)) != NULL)
    {
        llama_free_model(entry->model);
        hash_search(shared_models, &entry->model_name, HASH_REMOVE, NULL);
    }

    llama_backend_free();
}

Datum text_to_token_length(PG_FUNCTION_ARGS) {
    text *model_filename_text = PG_GETARG_TEXT_P(0);
    char* model_filename = text_to_cstring(model_filename_text);
    bool found;
    SharedModel *entry;

    elog(DEBUG1, "Acquiring lock to run prompt");
    LWLockAcquire(ModelHashLock, LW_EXCLUSIVE);

    entry = (SharedModel *) hash_search(shared_models, model_filename, HASH_ENTER, &found);
    if (!found) {
        LWLockRelease(ModelHashLock);
        ereport(ERROR, (errcode_for_file_access(), errmsg("No loaded model with name: %s", model_filename)));
    }

    text *prompt_text = PG_GETARG_TEXT_P(1);
    char* prompt = text_to_cstring(prompt_text);
    llama_context_params ctx_params = llama_context_default_params();
    // ctx_params.n_ctx = 0;
    ctx_params.n_ctx = 32768;
    ctx_params.rope_freq_scale = 1.0f / 16.0f;

    llama_context* ctx = llama_new_context_with_model(entry->model, ctx_params);

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, prompt, true);

    llama_free(ctx);

    LWLockRelease(ModelHashLock);

    PG_RETURN_INT32(tokens_list.size());
}

Datum text_to_tokens(PG_FUNCTION_ARGS) {
    elog(DEBUG1, "Tokenizing text.");
    text *model_filename_text = PG_GETARG_TEXT_P(0);
    char* model_filename = text_to_cstring(model_filename_text);
    bool found;
    SharedModel *entry;

    elog(DEBUG1, "Acquiring lock to run prompt");
    LWLockAcquire(ModelHashLock, LW_EXCLUSIVE);

    entry = (SharedModel *) hash_search(shared_models, model_filename, HASH_ENTER, &found);
    if (!found) {
        LWLockRelease(ModelHashLock);
        ereport(ERROR, (errcode_for_file_access(), errmsg("No loaded model with name: %s", model_filename)));
    }

    text *prompt_text = PG_GETARG_TEXT_P(1);
    char* prompt = text_to_cstring(prompt_text);
    llama_context_params ctx_params = llama_context_default_params();
    // ctx_params.n_ctx = 0;
    ctx_params.n_ctx = 32768;
    ctx_params.rope_freq_scale = 1.0f / 16.0f;

    llama_context* ctx = llama_new_context_with_model(entry->model, ctx_params);

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, prompt, true);

    llama_free(ctx);

    LWLockRelease(ModelHashLock);

    int nelems = tokens_list.size();
    llama_token *data = tokens_list.data();

    int ndim = 1;
    int dims[1] = { nelems };
    int lbs[1] = { 1 };

    ArrayType *result = construct_array((Datum*) data, nelems, INT4OID, sizeof(llama_token), true, 'i');

    elog(DEBUG1, "Tokenized text.");
    PG_RETURN_ARRAYTYPE_P(result);
}
Datum model_n_ctx(PG_FUNCTION_ARGS) {
    text *model_filename_text = PG_GETARG_TEXT_P(0);
    char* model_filename = text_to_cstring(model_filename_text);
    bool found;
    SharedModel *entry;

    elog(DEBUG1, "Acquiring lock to run prompt");
    LWLockAcquire(ModelHashLock, LW_EXCLUSIVE);

    entry = (SharedModel *) hash_search(shared_models, model_filename, HASH_ENTER, &found);
    if (!found) {
        LWLockRelease(ModelHashLock);
        ereport(ERROR, (errcode_for_file_access(), errmsg("No loaded model with name: %s", model_filename)));
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;

    llama_context* ctx = llama_new_context_with_model(entry->model, ctx_params);
    int n_ctx = llama_n_ctx(ctx);
    llama_free(ctx);
    LWLockRelease(ModelHashLock);
    PG_RETURN_INT32(n_ctx);
}

Datum load_model(PG_FUNCTION_ARGS) {
    text* model_filename_text = PG_GETARG_TEXT_P(0);
    char* model_filename = text_to_cstring(model_filename_text);

    bool found;
    SharedModel *entry;

    elog(DEBUG1, "Acquiring lock to load model");
    LWLockAcquire(ModelHashLock, LW_EXCLUSIVE);

    elog(DEBUG1, "Searching for model");
    entry = (SharedModel *) hash_search(shared_models, model_filename, HASH_ENTER, &found);
    if (!found) {
        llama_model_params model_params = llama_model_default_params();
        elog(DEBUG1, "Model not found: loading");
        entry->model = llama_load_model_from_file(model_filename, model_params);
        if (entry->model == NULL)
        {
            LWLockRelease(ModelHashLock);
            ereport(ERROR, (errcode_for_file_access(), errmsg("Failed to load model from file: %s", model_filename)));
        }
    }

    elog(DEBUG1, "Loaded model: releasing lock");
    LWLockRelease(ModelHashLock);

    PG_RETURN_VOID();
}

Datum prompt_model(PG_FUNCTION_ARGS) {
    text *model_filename_text = PG_GETARG_TEXT_P(0);
    char* model_filename = text_to_cstring(model_filename_text);
    bool found;
    SharedModel *entry;

    elog(DEBUG1, "Acquiring lock to run prompt");
    LWLockAcquire(ModelHashLock, LW_EXCLUSIVE);

    entry = (SharedModel *) hash_search(shared_models, model_filename, HASH_ENTER, &found);
    if (!found) {
        LWLockRelease(ModelHashLock);
        ereport(ERROR, (errcode_for_file_access(), errmsg("No loaded model with name: %s", model_filename)));
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.seed = 1234;
    int32 n_len = PG_GETARG_INT32(1);
    int32 max_ctx_len = PG_GETARG_INT32(2);
    ctx_params.n_ctx = n_len;
    ctx_params.n_batch = n_len;
    if (max_ctx_len < n_len) {
        ctx_params.rope_freq_scale = ((float) n_len) / ((float) max_ctx_len);
    }
    // ctx_params.n_threads = get_math_cpu_count();
    ctx_params.n_threads = 16;
    // ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;
    ctx_params.n_threads_batch = ctx_params.n_threads;
    // DEFAULT:
    // ctx_params.type_k = GGML_TYPE_F16;
    // COMPAT:
    // ctx_params.type_k = GGML_TYPE_Q4_K;

    llama_context* ctx = llama_new_context_with_model(entry->model, ctx_params);

    text *prompt_text = PG_GETARG_TEXT_P(3);
    char* prompt = text_to_cstring(prompt_text);
    std::vector<llama_token> tokens_list;
    elog(DEBUG1, "Tokenizing prompt text.");
    tokens_list = ::llama_tokenize(ctx, prompt, true);

    // elog(DEBUG1, "Loading suffix tokens.");
    // ArrayType *append_arr = PG_GETARG_ARRAYTYPE_P(2);
    // if (ARR_NULLBITMAP(append_arr)) {
    //     ereport(ERROR,
    //             (errmsg("null elements are not supported")));
    // }

    // int nelems = ArrayGetNItems(ARR_NDIM(append_arr), ARR_DIMS(append_arr));
    // llama_token *append_data = (llama_token*)ARR_DATA_PTR(append_arr);

    // elog(DEBUG1, "Converting suffix tokens to vec.");
    // std::vector<llama_token> append_tokens(append_data, append_data + nelems);

    // tokens_list.insert(tokens_list.end(), append_tokens.begin(), append_tokens.end());
    // size_t tokens_list_size = tokens_list.size();
    // size_t init_tokens_size = tokens_list_size + append_tokens.size();
    size_t init_tokens_size = tokens_list.size();

    llama_batch batch = llama_batch_init(init_tokens_size, 0, 1);
    elog(DEBUG1, "Loading text to batch.");
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }
    // elog(DEBUG1, "Loading suffix to batch.");
    // for (size_t i = 0; i < append_tokens.size(); i++) {
    //     llama_batch_add(batch, append_tokens[i], i + tokens_list_size, { 0 }, false);
    // }

    elog(DEBUG1, "Running batch.");
    batch.logits[batch.n_tokens - 1] = true;
    elog(DEBUG1, "Decoding initial batch.");
    if (llama_decode(ctx, batch) != 0) {
        LWLockRelease(ModelHashLock);
        ereport(ERROR, (errcode_for_file_access(),
            errmsg(
            "%s: llama_decode() failed\n", __func__
        )));
    }

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    std::string res;

    elog(DEBUG1, "Looping through batch.");
    while (n_cur <= n_len) {
        {
            // elog(DEBUG1, "Loading vocab.");
            auto   n_vocab = llama_n_vocab(entry->model);
            // elog(DEBUG1, "Loading logits.");
            auto * logits  = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // elog(DEBUG1, "Sampling token.");
            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // const llama_token new_token_id = llama_sampling_sample(ctx_sampling, ctx, NULL, n_cur);
            // llama_sampling_accept(ctx_sampling, ctx, new_token_id, true);

            if (llama_token_is_eog(entry->model, new_token_id) || n_cur == n_len) {
                res += "\n";

                break;
            }

            // elog(DEBUG1, "Adding piece.");
            res += llama_token_to_piece(ctx, new_token_id);

            // elog(DEBUG1, "Clearing batch.");
            llama_batch_clear(batch);

            // elog(DEBUG1, "Adding batch.");
            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

            n_decode += 1;
        }

        n_cur += 1;

        // elog(DEBUG1, "Decoding batch.");
        if (llama_decode(ctx, batch)) {
            LWLockRelease(ModelHashLock);
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

    }

    text *res_text = cstring_to_text((char *) res.c_str());

    llama_batch_free(batch);
    llama_free(ctx);

    LWLockRelease(ModelHashLock);

    PG_RETURN_TEXT_P(res_text);
}

Datum free_model(PG_FUNCTION_ARGS) {
    text *model_filename_text = PG_GETARG_TEXT_P(0);
    char *model_filename = text_to_cstring(model_filename_text);
    HASH_SEQ_STATUS status;
    SharedModel *entry;
    bool found;

    entry = (SharedModel *) hash_search(shared_models, model_filename, HASH_ENTER, &found);

    if (found) {
        llama_free_model(entry->model);
        hash_search(shared_models, &entry->model_name, HASH_REMOVE, NULL);
    } else {
        ereport(ERROR, (errmsg("Model not found in shared memory")));
    }

    PG_RETURN_VOID();
}