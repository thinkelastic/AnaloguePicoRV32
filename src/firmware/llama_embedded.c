/*
 * Llama-2 Inference for PicoRV32 on Analogue Pocket
 *
 * This is an embedded adaptation of llama2.c (karpathy/llama2.c)
 * Modified to work with:
 *   - Data slot loading instead of file I/O
 *   - SDRAM for model weights and heap
 *   - Terminal output instead of stdout
 *
 * Original: https://github.com/karpathy/llama2.c
 */

#include "libc/libc.h"
#include "dataslot.h"
#include "terminal.h"

/* Redirect printf to terminal */
#define printf term_printf

/* Configuration - adjust these for your model */
#define DEFAULT_STEPS       64      /* Max tokens to generate */
#define DEFAULT_TEMPERATURE 0.8f
#define DEFAULT_TOPP        0.9f
#define DEFAULT_PROMPT      "Once upon a time"

/* ============================================
 * Transformer model structures
 * ============================================ */

typedef struct {
    int dim;         /* Transformer dimension */
    int hidden_dim;  /* FFN hidden dimension */
    int n_layers;    /* Number of layers */
    int n_heads;     /* Number of attention heads */
    int n_kv_heads;  /* Number of KV heads (can be < n_heads for MQA) */
    int vocab_size;  /* Vocabulary size */
    int seq_len;     /* Max sequence length */
} Config;

typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float* key_cache;
    float* value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    float* data;
    size_t file_size;
} Transformer;

/* ============================================
 * Tokenizer structures
 * ============================================ */

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

/* ============================================
 * Sampler structures
 * ============================================ */

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

/* ============================================
 * Memory allocation for run state
 * ============================================ */

static void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        printf("ERROR: malloc failed for run state!\n");
        while(1);
    }
}

static void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

/* ============================================
 * Weight memory mapping
 * ============================================ */

static void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;

    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; /* skip freq_cis_real */
    ptr += p->seq_len * head_size / 2; /* skip freq_cis_imag */
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

/* ============================================
 * Build transformer from SDRAM data
 * ============================================ */

static void build_transformer_from_memory(Transformer *t, void* data, size_t size) {
    printf("  Reading config from SDRAM...\n");
    Config* config = (Config*)data;

    printf("  Config ptr: 0x%08X\n", (uint32_t)config);
    printf("  Reading dim...\n");
    int dim = config->dim;
    printf("  dim=%d\n", dim);

    t->config = *config;

    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    t->config.vocab_size = abs(config->vocab_size);

    printf("  vocab_size=%d, shared=%d\n", t->config.vocab_size, shared_weights);

    float* weights_ptr = (float*)((char*)data + sizeof(Config));
    printf("  Mapping weights at 0x%08X...\n", (uint32_t)weights_ptr);
    memory_map_weights(&t->weights, &t->config, weights_ptr, shared_weights);

    printf("  Allocating run state...\n");
    malloc_run_state(&t->state, &t->config);

    t->data = data;
    t->file_size = size;
    printf("  Transformer built.\n");
}

static void free_transformer(Transformer* t) {
    free_run_state(&t->state);
    /* Don't free t->data - it's in SDRAM and managed elsewhere */
}

/* ============================================
 * Neural network operations
 * ============================================ */

static void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

static void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

static void matmul(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

static float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));

    for (unsigned long long l = 0; l < (unsigned long long)p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

/* ============================================
 * Tokenizer
 * ============================================ */

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

static void build_tokenizer_from_memory(Tokenizer* t, void* data, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;

    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    uint8_t* ptr = (uint8_t*)data;

    t->max_token_length = *((int*)ptr);
    ptr += sizeof(int);

    for (int i = 0; i < vocab_size; i++) {
        t->vocab_scores[i] = *((float*)ptr);
        ptr += sizeof(float);

        int len = *((int*)ptr);
        ptr += sizeof(int);

        t->vocab[i] = (char*)malloc(len + 1);
        memcpy(t->vocab[i], ptr, len);
        t->vocab[i][len] = '\0';
        ptr += len;
    }
}

static void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    if (t->sorted_vocab) free(t->sorted_vocab);
}

static char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

static void safe_printf(char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

static void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) {
        printf("ERROR: cannot encode NULL text\n");
        while(1);
    }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = malloc((t->max_token_length*2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;

    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;
        }

        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

/* ============================================
 * Sampler
 * ============================================ */

static int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

static int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

static int compare_prob(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

static int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_prob);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

static void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

static void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

static float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q = 0; q < sampler->vocab_size; q++) {
            logits[q] /= sampler->temperature;
        }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

/* ============================================
 * Generation loop
 * ============================================ */

static void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        printf("ERROR: expected at least 1 prompt token\n");
        while(1);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) { break; }

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        token = next;

        if (start == 0) { start = time(NULL); }
    }
    printf("\n");

    if (pos > 1) {
        long end = time(NULL);
        if (end > start) {
            printf("Speed: %d tok/s\n", (int)((pos-1) / (end-start)));
        }
    }

    free(prompt_tokens);
}

/* ============================================
 * Main entry point
 * ============================================ */

/*
 * SDRAM Layout (64MB total):
 *
 * Bridge addresses (in data.json):     CPU addresses:
 * 0x00000000 ─────────────────────────► 0x10000000
 * │           MODEL DATA (~32MB max)        │
 * │     (APF auto-loads Slot 0 here)        │
 * │         stories15M.bin                  │
 * 0x02000000 ─────────────────────────► 0x12000000
 * │        TOKENIZER DATA (~1MB)            │
 * │     (APF auto-loads Slot 1 here)        │
 * │         tokenizer.bin (~424KB)          │
 * 0x02100000 ─────────────────────────► 0x12100000
 * │              HEAP ARENA                 │
 * │        (malloc/calloc/free)             │
 * │           ~31MB available               │
 * 0x03FFFFFF ─────────────────────────► 0x13FFFFFF
 *
 * APF loads to bridge addresses 0x00xxxxxx (from data.json).
 * CPU accesses SDRAM at 0x10xxxxxx (offset by 0x10000000).
 */
#define MODEL_SDRAM_ADDR      0x10000000                  /* Slot 0: bridge 0x00000000 */
#define TOKENIZER_SDRAM_ADDR  0x12000000                  /* Slot 1: bridge 0x02000000 */
#define HEAP_SDRAM_ADDR       0x12100000                  /* Heap start (after tokenizer) */
#define HEAP_SIZE             (0x14000000 - HEAP_SDRAM_ADDR)  /* ~31MB for heap */

void llama_main(void) {
    printf("llama2.c for Analogue Pocket\n");
    printf("============================\n\n");

    /* Wait for SDRAM to be ready */
    printf("Waiting for SDRAM...\n");
    while (!(SYS_STATUS & SYS_STATUS_SDRAM_READY)) {
        /* Busy wait */
    }
    printf("SDRAM ready.\n");

    /* Wait for APF to finish loading all data slots */
    printf("Waiting for data slots...\n");
    if (dataslot_wait_ready() != 0) {
        printf("  ERROR: Timeout waiting for data slots!\n");
        while(1);
    }
    printf("Data slots loaded.\n");

    /* Simple SDRAM / data verification */
    printf("Verifying model data...\n");
    volatile uint32_t *model_header = (volatile uint32_t *)MODEL_SDRAM_ADDR;
    uint32_t dim_check = model_header[0];  /* First field is 'dim' */
    printf("  Model header[0] (dim): %d\n", dim_check);

    if (dim_check == 0 || dim_check > 10000) {
        printf("  ERROR: Invalid model data!\n");
        printf("  (Check that model.bin is in /Assets/homebrew/common/)\n");
        while(1);
    }
    printf("  Model data verified!\n");

    /* Test SDRAM write/read from CPU */
    printf("Testing SDRAM write...\n");
    volatile uint32_t *test_addr = (volatile uint32_t *)HEAP_SDRAM_ADDR;
    uint32_t test_val = 0xDEADBEEF;
    *test_addr = test_val;
    uint32_t read_back = *test_addr;
    printf("  Wrote 0x%08X, read back 0x%08X\n", test_val, read_back);
    if (read_back != test_val) {
        printf("  ERROR: SDRAM write failed!\n");
        while(1);
    }
    printf("  SDRAM write OK!\n");

    /* Initialize heap in SDRAM (after model and tokenizer regions) */
    printf("Initializing heap at 0x%08X...\n", HEAP_SDRAM_ADDR);
    heap_init((void*)HEAP_SDRAM_ADDR, HEAP_SIZE);

    /* Data is now in SDRAM */
    void* model_data = (void*)MODEL_SDRAM_ADDR;
    void* tokenizer_data = (void*)TOKENIZER_SDRAM_ADDR;

    printf("Model at 0x%08X\n", (uint32_t)model_data);
    printf("Tokenizer at 0x%08X\n", (uint32_t)tokenizer_data);

    /* Build transformer from loaded data */
    printf("Building transformer...\n");
    Transformer transformer;
    build_transformer_from_memory(&transformer, model_data, 0);  /* size not needed */

    printf("Config: dim=%d, hidden=%d, layers=%d, heads=%d, vocab=%d, seq=%d\n",
           transformer.config.dim,
           transformer.config.hidden_dim,
           transformer.config.n_layers,
           transformer.config.n_heads,
           transformer.config.vocab_size,
           transformer.config.seq_len);

    /* Build tokenizer */
    printf("Building tokenizer...\n");
    Tokenizer tokenizer;
    build_tokenizer_from_memory(&tokenizer, tokenizer_data, transformer.config.vocab_size);

    /* Build sampler */
    printf("Building sampler...\n");
    Sampler sampler;
    unsigned long long seed = SYS_CYCLE_LO;  /* Use cycle counter as seed */
    build_sampler(&sampler, transformer.config.vocab_size, DEFAULT_TEMPERATURE, DEFAULT_TOPP, seed);

    /* Run generation */
    printf("\n--- Generating ---\n");
    printf("Prompt: \"%s\"\n\n", DEFAULT_PROMPT);

    generate(&transformer, &tokenizer, &sampler, (char*)DEFAULT_PROMPT, DEFAULT_STEPS);

    /* Cleanup */
    printf("\nCleaning up...\n");
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    printf("Done!\n");

    /* Halt */
    while(1);
}
