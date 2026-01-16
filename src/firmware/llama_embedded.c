/*
 * Llama-2 Inference for PicoRV32 on Analogue Pocket
 *
 * Clean adaptation of llama2.c (karpathy/llama2.c)
 * Modified to work with:
 *   - Data slot loading instead of file I/O
 *   - SDRAM for model weights (memory-mapped directly)
 *   - Simple heap for RunState buffers
 *
 * Original: https://github.com/karpathy/llama2.c
 */

#include "libc/libc.h"
#include "dataslot.h"
#include "terminal.h"

/* Redirect printf to terminal */
#define printf term_printf

/* Configuration */
#define DEFAULT_STEPS       64
#define DEFAULT_TEMPERATURE 0.0f   /* Use argmax for debugging */
#define DEFAULT_TOPP        0.9f
#define DEFAULT_PROMPT      "Once upon a time"

/* Memory map */
#define MODEL_ADDR          0x10000000   /* Slot 0: model.bin in SDRAM */
#define TOKENIZER_ADDR      0x12000000   /* Slot 1: tokenizer.bin in SDRAM */
#define PSRAM_HEAP_ADDR     0x14000000   /* PSRAM base for heap/working memory */
#define PSRAM_HEAP_SIZE     0x04000000   /* 64MB PSRAM available */

/* Use PSRAM for heap instead of SDRAM */
#define HEAP_ADDR           PSRAM_HEAP_ADDR
#define HEAP_SIZE           PSRAM_HEAP_SIZE

/* ============================================
 * Transformer model structures
 * ============================================ */

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
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
 * RunState allocation
 * ============================================ */

static void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    printf("  kv_dim=%d\n", kv_dim);

    /* Use malloc instead of calloc - calloc does memset which is very slow on SDRAM */
    s->x = malloc(p->dim * sizeof(float));
    s->xb = malloc(p->dim * sizeof(float));
    s->xb2 = malloc(p->dim * sizeof(float));
    s->hb = malloc(p->hidden_dim * sizeof(float));
    s->hb2 = malloc(p->hidden_dim * sizeof(float));
    s->q = malloc(p->dim * sizeof(float));

    int kv_size = p->n_layers * p->seq_len * kv_dim;
    printf("  kv_cache size: %d floats (%d KB)\n", kv_size, kv_size * 4 / 1024);
    s->key_cache = malloc(kv_size * sizeof(float));
    s->value_cache = malloc(kv_size * sizeof(float));

    s->att = malloc(p->n_heads * p->seq_len * sizeof(float));
    s->logits = malloc(p->vocab_size * sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        printf("malloc failed!\n");
        printf("  x=%p xb=%p xb2=%p\n", s->x, s->xb, s->xb2);
        printf("  hb=%p hb2=%p q=%p\n", s->hb, s->hb2, s->q);
        printf("  key=%p val=%p\n", s->key_cache, s->value_cache);
        printf("  att=%p logits=%p\n", s->att, s->logits);
        while(1);
    }
    printf("  RunState allocated OK\n");
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
 * Build transformer from SDRAM
 * ============================================ */

static void build_transformer(Transformer *t, void* data) {
    Config* config_ptr = (Config*)data;
    t->config = *config_ptr;
    int shared_weights = t->config.vocab_size > 0 ? 1 : 0;
    t->config.vocab_size = abs(t->config.vocab_size);

    float* weights_ptr = (float*)((char*)data + sizeof(Config));
    memory_map_weights(&t->weights, &t->config, weights_ptr, shared_weights);
    malloc_run_state(&t->state, &t->config);
    t->data = data;
}

static void free_transformer(Transformer* t) {
    free_run_state(&t->state);
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
        if (x[i] > max_val) max_val = x[i];
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

static void build_tokenizer(Tokenizer* t, void* data, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;  /* Built lazily in encode() */

    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    /* Read tokenizer from SDRAM */
    uint8_t* ptr = (uint8_t*)data;

    /* Read max_token_length */
    memcpy(&t->max_token_length, ptr, sizeof(int));
    ptr += sizeof(int);

    printf("max_token_length=%d\n", t->max_token_length);

    /* Read each token */
    for (int i = 0; i < vocab_size; i++) {
        /* Read score */
        memcpy(&t->vocab_scores[i], ptr, sizeof(float));
        ptr += sizeof(float);

        /* Read length */
        int len;
        memcpy(&len, ptr, sizeof(int));
        ptr += sizeof(int);

        /* Allocate and read string */
        t->vocab[i] = (char*)malloc(len + 1);
        if (!t->vocab[i]) {
            printf("malloc failed for token %d\n", i);
            while(1);
        }
        memcpy(t->vocab[i], ptr, len);
        t->vocab[i][len] = '\0';
        ptr += len;
    }

    printf("Loaded %d tokens\n", vocab_size);
}

static void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

static char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    /* Handle byte tokens like <0x0A> */
    unsigned char byte_val;
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x') {
        /* Parse hex value */
        byte_val = 0;
        for (int i = 3; piece[i] != '>' && piece[i] != '\0'; i++) {
            char c = piece[i];
            if (c >= '0' && c <= '9') byte_val = (byte_val << 4) | (c - '0');
            else if (c >= 'a' && c <= 'f') byte_val = (byte_val << 4) | (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') byte_val = (byte_val << 4) | (c - 'A' + 10);
        }
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

static void safe_printf(char *piece) {
    if (piece == NULL || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) return;
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
        printf("cannot encode NULL text\n");
        while(1);
    }

    /* Lazily build sorted_vocab on first encode */
    if (t->sorted_vocab == NULL) {
        printf("Sorting vocabulary...\n");
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
        printf("Sort complete\n");
    }

    char* str_buffer = malloc((t->max_token_length * 2 + 3) * sizeof(char));
    size_t str_len = 0;

    *n_tokens = 0;

    if (bos) {
        tokens[(*n_tokens)++] = 1;
    }

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

    /* BPE merge loop */
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            /* Build merged string using sprintf equivalent */
            char *tok1 = t->vocab[tokens[i]];
            char *tok2 = t->vocab[tokens[i+1]];
            int len1 = strlen(tok1);
            int len2 = strlen(tok2);
            memcpy(str_buffer, tok1, len1);
            memcpy(str_buffer + len1, tok2, len2);
            str_buffer[len1 + len2] = '\0';

            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) break;

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;
    }

    if (eos) {
        tokens[(*n_tokens)++] = 2;
    }

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
        if (coin < cdf) return i;
    }
    return n - 1;
}

static int compare_prob(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*)a;
    ProbIndex* b_ = (ProbIndex*)b;
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
        if (r < cdf) return probindex[i].index;
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
    if (prompt == NULL) prompt = empty_prompt;

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    printf("Encoded %d tokens:", num_prompt_tokens);
    for (int i = 0; i < num_prompt_tokens && i < 10; i++) {
        printf(" %d", prompt_tokens[i]);
    }
    printf("\n");

    if (num_prompt_tokens < 1) {
        printf("expected at least 1 prompt token\n");
        while(1);
    }

    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    printf("\n");
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) break;

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        token = next;
    }
    printf("\n");

    free(prompt_tokens);
}

/* ============================================
 * Main entry point
 * ============================================ */

/* ============================================
 * Comprehensive Memory Test
 * ============================================ */

static void memory_test(void) {
    printf("=== SDRAM Memory Test ===\n\n");

    volatile uint32_t* sdram = (volatile uint32_t*)MODEL_ADDR;
    uint32_t errors = 0;
    uint32_t tests = 0;

    /* Test 1: Basic word write/read */
    printf("Test 1: Basic word write/read\n");
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t val = 0xDEAD0000 | i;
        volatile uint32_t* ptr = (volatile uint32_t*)((uintptr_t)HEAP_ADDR + i * 4);
        *ptr = val;
        uint32_t read = *ptr;
        tests++;
        if (read != val) {
            printf("  FAIL @%d: wrote %08X, read %08X\n", i, val, read);
            errors++;
        }
    }
    printf("  %d/%d passed\n", tests - errors, tests);

    /* Test 2: Walking ones */
    printf("Test 2: Walking ones pattern\n");
    volatile uint32_t* test_addr = (volatile uint32_t*)(HEAP_ADDR + 0x100);
    uint32_t walk_errors = 0;
    for (int bit = 0; bit < 32; bit++) {
        uint32_t pattern = 1 << bit;
        *test_addr = pattern;
        uint32_t read = *test_addr;
        tests++;
        if (read != pattern) {
            printf("  FAIL bit %d: wrote %08X, read %08X\n", bit, pattern, read);
            walk_errors++;
            errors++;
        }
    }
    printf("  %d/32 passed\n", 32 - walk_errors);

    /* Test 3: Walking zeros */
    printf("Test 3: Walking zeros pattern\n");
    uint32_t walk0_errors = 0;
    for (int bit = 0; bit < 32; bit++) {
        uint32_t pattern = ~(1 << bit);
        *test_addr = pattern;
        uint32_t read = *test_addr;
        tests++;
        if (read != pattern) {
            printf("  FAIL bit %d: wrote %08X, read %08X\n", bit, pattern, read);
            walk0_errors++;
            errors++;
        }
    }
    printf("  %d/32 passed\n", 32 - walk0_errors);

    /* Test 4: Address uniqueness - write unique values */
    printf("Test 4: Address uniqueness (256 words)\n");
    uint32_t addr_errors = 0;
    volatile uint32_t* base = (volatile uint32_t*)(HEAP_ADDR + 0x200);
    for (int i = 0; i < 256; i++) {
        base[i] = 0xA5000000 | i;
    }
    for (int i = 0; i < 256; i++) {
        uint32_t expected = 0xA5000000 | i;
        uint32_t read = base[i];
        tests++;
        if (read != expected) {
            if (addr_errors < 5) {
                printf("  FAIL @%d: expected %08X, read %08X\n", i, expected, read);
            }
            addr_errors++;
            errors++;
        }
    }
    printf("  %d/256 passed\n", 256 - addr_errors);

    /* Test 5: Byte ordering test */
    printf("Test 5: Byte ordering\n");
    volatile uint32_t* byte_test = (volatile uint32_t*)(HEAP_ADDR + 0x800);
    *byte_test = 0x04030201;
    uint32_t byte_read = *byte_test;
    volatile uint8_t* bytes = (volatile uint8_t*)byte_test;
    printf("  Wrote: 0x04030201\n");
    printf("  Read:  0x%08X\n", byte_read);
    printf("  Bytes: [0]=%02X [1]=%02X [2]=%02X [3]=%02X\n",
           bytes[0], bytes[1], bytes[2], bytes[3]);
    tests++;
    if (byte_read != 0x04030201) {
        printf("  FAIL: byte order mismatch!\n");
        errors++;
    }

    /* Test 6: Sequential burst write/read */
    printf("Test 6: Sequential burst (1KB)\n");
    uint32_t burst_errors = 0;
    volatile uint32_t* burst = (volatile uint32_t*)(HEAP_ADDR + 0x1000);
    for (int i = 0; i < 256; i++) {
        burst[i] = i * 0x01010101;
    }
    for (int i = 0; i < 256; i++) {
        uint32_t expected = i * 0x01010101;
        uint32_t read = burst[i];
        tests++;
        if (read != expected) {
            if (burst_errors < 5) {
                printf("  FAIL @%d: expected %08X, read %08X\n", i, expected, read);
            }
            burst_errors++;
            errors++;
        }
    }
    printf("  %d/256 passed\n", 256 - burst_errors);

    /* Test 7: Random pattern */
    printf("Test 7: Pseudo-random pattern (256 words)\n");
    uint32_t rand_errors = 0;
    volatile uint32_t* rand_base = (volatile uint32_t*)(HEAP_ADDR + 0x2000);
    uint32_t seed = 0x12345678;
    /* Write */
    uint32_t r = seed;
    for (int i = 0; i < 256; i++) {
        r ^= r << 13;
        r ^= r >> 17;
        r ^= r << 5;
        rand_base[i] = r;
    }
    /* Read and verify */
    r = seed;
    for (int i = 0; i < 256; i++) {
        r ^= r << 13;
        r ^= r >> 17;
        r ^= r << 5;
        uint32_t read = rand_base[i];
        tests++;
        if (read != r) {
            if (rand_errors < 5) {
                printf("  FAIL @%d: expected %08X, read %08X\n", i, r, read);
            }
            rand_errors++;
            errors++;
        }
    }
    printf("  %d/256 passed\n", 256 - rand_errors);

    /* Test 8: Heap allocator test */
    printf("Test 8: Heap allocator\n");
    heap_init((void*)HEAP_ADDR, HEAP_SIZE);
    int heap_errors = 0;

    void* p1 = malloc(64);
    void* p2 = malloc(128);
    void* p3 = malloc(256);
    printf("  malloc(64)  = %p\n", p1);
    printf("  malloc(128) = %p\n", p2);
    printf("  malloc(256) = %p\n", p3);

    if (!p1 || !p2 || !p3) {
        printf("  FAIL: malloc returned NULL\n");
        heap_errors++;
    } else {
        /* Write to allocated memory */
        memset(p1, 0xAA, 64);
        memset(p2, 0xBB, 128);
        memset(p3, 0xCC, 256);

        /* Verify */
        uint8_t* b1 = (uint8_t*)p1;
        uint8_t* b2 = (uint8_t*)p2;
        uint8_t* b3 = (uint8_t*)p3;

        int ok = 1;
        for (int i = 0; i < 64; i++) if (b1[i] != 0xAA) ok = 0;
        for (int i = 0; i < 128; i++) if (b2[i] != 0xBB) ok = 0;
        for (int i = 0; i < 256; i++) if (b3[i] != 0xCC) ok = 0;

        if (!ok) {
            printf("  FAIL: memory corruption detected\n");
            heap_errors++;
        } else {
            printf("  Write/read OK\n");
        }

        free(p1);
        free(p2);
        free(p3);

        /* Test reallocation */
        void* p4 = malloc(64);
        printf("  After free, malloc(64) = %p\n", p4);
        if (!p4) {
            printf("  FAIL: malloc after free failed\n");
            heap_errors++;
        }
    }
    tests += 4;
    errors += heap_errors;

    /* Test 9: Large allocation */
    printf("Test 9: Large allocation (1MB)\n");
    heap_init((void*)HEAP_ADDR, HEAP_SIZE);  /* Re-init heap */
    void* big = malloc(1024 * 1024);
    printf("  malloc(1MB) = %p\n", big);
    if (!big) {
        printf("  FAIL: large allocation failed\n");
        errors++;
    } else {
        /* Write pattern to start, middle, end */
        volatile uint32_t* bp = (volatile uint32_t*)big;
        bp[0] = 0x11111111;
        bp[1024*128] = 0x22222222;  /* Middle (512KB / 4) */
        bp[1024*256 - 1] = 0x33333333;  /* End (1MB / 4 - 1) */

        uint32_t r0 = bp[0];
        uint32_t r1 = bp[1024*128];
        uint32_t r2 = bp[1024*256 - 1];

        printf("  Start:  wrote 11111111, read %08X\n", r0);
        printf("  Middle: wrote 22222222, read %08X\n", r1);
        printf("  End:    wrote 33333333, read %08X\n", r2);

        if (r0 != 0x11111111 || r1 != 0x22222222 || r2 != 0x33333333) {
            printf("  FAIL: data mismatch\n");
            errors++;
        }
    }
    tests++;

    /* Summary */
    printf("\n=== Summary ===\n");
    printf("Total tests: %d\n", tests);
    printf("Errors: %d\n", errors);
    if (errors == 0) {
        printf("ALL TESTS PASSED!\n");
    } else {
        printf("SOME TESTS FAILED!\n");
    }
}

void llama_main(void) {
    printf("PicoRV32 Memory Test\n");
    printf("====================\n\n");

    /* Wait for SDRAM */
    printf("Waiting for SDRAM...\n");
    while (!(SYS_STATUS & SYS_STATUS_SDRAM_READY));
    printf("SDRAM ready.\n");

    /* Wait for data slots */
    printf("Waiting for data slots...\n");
    if (dataslot_wait_ready() != 0) {
        printf("Timeout waiting for data slots!\n");
        while(1);
    }
    printf("Data slots loaded.\n\n");

    /* Run memory test */
    memory_test();

    printf("\nTest complete. Halting.\n");
    while(1);
}
