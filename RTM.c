
// please see the ERS Github repo for further implementation 
/*
 * Double_RTM.c
 *
 * A small, portable illustrative implementation of a "double recursive"
 * processing loop inspired by the Recursive Transformer Model (RTM)
 * and intended to demonstrate how ERS-style memory + reconsideration loops
 * might be prototyped in plain C.
 *
 * This code is intentionally simple and lightweight: it uses deterministic,
 * pseudo-embeddings (no external ML libs) and a small persistent memory
 * representation on disk (text file). It demonstrates:
 *  - MemoryBlock structures (text, embedding, confidence, timestamp)
 *  - Adding memory, saving/loading persistent memory
 *  - A two-pass "double" recursive reconsideration loop:
 *      pass 1: local consensus and embedding refinement
 *      pass 2: contradiction detection and optional rewrite (blend)
 *
 * Citation:
 * Josef Kurk Edwards. The Recursive Transformer Model: Architecture,
 * Theory, and Implementation with Persistent Memory Logic Loops.
 * TechRxiv, Oct 23, 2025.
 * DOI: 10.36227/techrxiv.176118936.69886233/v1
 *
 * Funder: U.S. Department of Defense (Identifier: 100000005)
 *
 * Build:
 *   cc -O2 -o Double_RTM Double_RTM.c -lm
 *
 * Run:
 *   ./Double_RTM
 *
 * Note:
 * - This is an illustrative toy; replace pseudo_embedding() with real models
 *   (sentence-transformers, torch) when moving to a production ERS pipeline.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define EMB_DIM 32
#define MEMORY_SLOTS 16
#define MAX_TEXT 256
#define MEMORY_FILE "double_rtm_memory.txt"

typedef struct {
    char text[MAX_TEXT];
    float embedding[EMB_DIM];
    float confidence;   /* 0.0 .. 1.0 */
    long created_at;
} MemoryBlock;

/* --- Utility vector ops --- */
static void normalize(float *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += v[i] * v[i];
    s = sqrt(s);
    if (s < 1e-12) return;
    for (int i = 0; i < n; ++i) v[i] /= (float)s;
}

static float dotp(const float *a, const float *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)a[i] * (double)b[i];
    return (float)s;
}

static void add_scaled(float *dst, const float *src, int n, float scale) {
    for (int i = 0; i < n; ++i) dst[i] += src[i] * scale;
}

/* --- Deterministic pseudo-embedding: simple char-based hashing into a vector --- */
static void pseudo_embedding(const char *text, float *emb_out) {
    /* Initialize with small constant pattern to avoid zero vectors */
    for (int i = 0; i < EMB_DIM; ++i) emb_out[i] = (float)(0.01 * (i + 1));
    unsigned int state = 2166136261u;
    const unsigned char *p = (const unsigned char *)text;
    while (*p) {
        state ^= (unsigned int)(*p);
        state *= 16777619u;
        for (int i = 0; i < EMB_DIM; ++i) {
            /* mix state into vector deterministically */
            float t = (float)(((state >> (i % 24)) & 0xFF) - 128) / 128.0f;
            emb_out[i] += sinf((i+1) * 0.37f + t * 0.13f);
        }
        ++p;
    }
    normalize(emb_out, EMB_DIM);
}

/* --- Persistence: save/load memory to a simple text file --- */
static int save_memory(const char *filename, MemoryBlock *mem, int n_slots) {
    FILE *f = fopen(filename, "w");
    if (!f) return 0;
    for (int i = 0; i < n_slots; ++i) {
        if (mem[i].text[0] == '\0') continue;
        fprintf(f, "BEGIN_BLOCK\n");
        fprintf(f, "text:%s\n", mem[i].text);
        fprintf(f, "confidence:%g\n", mem[i].confidence);
        fprintf(f, "created_at:%ld\n", mem[i].created_at);
        fprintf(f, "embedding:");
        for (int j = 0; j < EMB_DIM; ++j) {
            fprintf(f, "%g%s", mem[i].embedding[j], (j+1==EMB_DIM) ? "\n" : ",");
        }
        fprintf(f, "END_BLOCK\n");
    }
    fclose(f);
    return 1;
}

static int load_memory(const char *filename, MemoryBlock *mem, int max_slots) {
    FILE *f = fopen(filename, "r");
    if (!f) return 0;
    char line[2048];
    MemoryBlock tmp;
    int slots = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "BEGIN_BLOCK", 11) == 0) {
            tmp.text[0] = '\0';
            tmp.confidence = 0.5f;
            tmp.created_at = time(NULL);
            for (int i = 0; i < EMB_DIM; ++i) tmp.embedding[i] = 0.0f;
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "END_BLOCK", 9) == 0) {
                    if (slots < max_slots) mem[slots++] = tmp;
                    break;
                } else if (strncmp(line, "text:", 5) == 0) {
                    strncpy(tmp.text, line + 5, MAX_TEXT - 1);
                    /* strip newline */
                    tmp.text[strcspn(tmp.text, "\r\n")] = 0;
                } else if (strncmp(line, "confidence:", 11) == 0) {
                    tmp.confidence = (float)atof(line + 11);
                } else if (strncmp(line, "created_at:", 11) == 0) {
                    tmp.created_at = atol(line + 11);
                } else if (strncmp(line, "embedding:", 10) == 0) {
                    char *p = line + 10;
                    for (int j = 0; j < EMB_DIM; ++j) {
                        char *comma = strchr(p, ',');
                        if (comma) {
                            *comma = '\0';
                            tmp.embedding[j] = (float)atof(p);
                            p = comma + 1;
                        } else {
                            tmp.embedding[j] = (float)atof(p);
                            break;
                        }
                    }
                    normalize(tmp.embedding, EMB_DIM);
                }
            }
        }
    }
    fclose(f);
    return slots;
}

/* --- Memory management: simple circular push (older entries dropped) --- */
static void push_memory(MemoryBlock *mem, int *n_slots, int max_slots, const char *text, float init_confidence) {
    if (*n_slots < max_slots) {
        strncpy(mem[*n_slots].text, text, MAX_TEXT - 1);
        mem[*n_slots].text[MAX_TEXT-1] = '\0';
        pseudo_embedding(text, mem[*n_slots].embedding);
        mem[*n_slots].confidence = init_confidence;
        mem[*n_slots].created_at = time(NULL);
        (*n_slots)++;
    } else {
        /* overwrite oldest (slot 0), shift left */
        for (int i = 0; i < max_slots - 1; ++i) mem[i] = mem[i+1];
        strncpy(mem[max_slots-1].text, text, MAX_TEXT - 1);
        mem[max_slots-1].text[MAX_TEXT-1] = '\0';
        pseudo_embedding(text, mem[max_slots-1].embedding);
        mem[max_slots-1].confidence = init_confidence;
        mem[max_slots-1].created_at = time(NULL);
    }
}

/* --- Find nearest neighbour index by cosine similarity --- */
static int find_nearest(MemoryBlock *mem, int n_slots, float *query_emb, float *best_sim) {
    int best = -1;
    float bests = -2.0f;
    for (int i = 0; i < n_slots; ++i) {
        float s = dotp(mem[i].embedding, query_emb, EMB_DIM);
        if (s > bests) {
            bests = s;
            best = i;
        }
    }
    if (best_sim) *best_sim = bests;
    return best;
}

/* --- Reconsideration flow: decay -> consensus -> contradiction -> optional rewrite --- */
static void reconsider_pass(MemoryBlock *mem, int n_slots, int pass_num) {
    /* Parameters (these would be configurable in ERS) */
    const float decay_rate = 0.97f;         /* confidence decay per pass */
    const float contradiction_threshold = 0.55f; /* cosine below which we consider contradiction */
    const float rewrite_blend = 0.4f;       /* how much of consensus to mix into text embedding */
    printf("=== Reconsideration pass %d (n_slots=%d) ===\n", pass_num, n_slots);

    for (int i = 0; i < n_slots; ++i) {
        /* decay confidence */
        mem[i].confidence *= decay_rate;

        /* find nearest neighbor (excluding self) */
        float best_sim = -2.0f;
        int best = -1;
        for (int j = 0; j < n_slots; ++j) {
            if (j == i) continue;
            float s = dotp(mem[i].embedding, mem[j].embedding, EMB_DIM);
            if (s > best_sim) { best_sim = s; best = j; }
        }

        if (best < 0) continue;

        /* compute a simple consensus embedding: weighted average of neighbor(s) */
        float consensus[EMB_DIM];
        for (int k = 0; k < EMB_DIM; ++k) consensus[k] = 0.0f;
        /* here we just take the single best neighbor for simplicity */
        add_scaled(consensus, mem[best].embedding, EMB_DIM, 1.0f);
        normalize(consensus, EMB_DIM);

        /* compute contradiction score (low cosine -> contradiction) */
        float sim = dotp(mem[i].embedding, consensus, EMB_DIM);

        printf("slot %d: conf=%.3f nearest=%d sim=%.4f\n", i, mem[i].confidence, best, sim);

        if (sim < contradiction_threshold) {
            /* contradiction detected: propose a rewrite by blending embeddings */
            printf("  -> contradiction detected (sim %.4f < %.3f), proposing rewrite\n", sim, contradiction_threshold);
            /* Blend current embedding towards consensus */
            for (int k = 0; k < EMB_DIM; ++k) {
                mem[i].embedding[k] = (1.0f - rewrite_blend) * mem[i].embedding[k] + rewrite_blend * consensus[k];
            }
            normalize(mem[i].embedding, EMB_DIM);
            /* Increase confidence slightly after rewrite (simulated acceptance) */
            mem[i].confidence = fminf(1.0f, mem[i].confidence + 0.08f);
            /* Optionally, rewrite textual content (here we append a tag) */
            size_t tl = strnlen(mem[i].text, MAX_TEXT-1);
            if (tl + 20 < MAX_TEXT-1) {
                strncat(mem[i].text, " [rewritten]", MAX_TEXT-1 - tl);
            }
        } else {
            /* No contradiction: refine embedding slightly toward consensus (consensus reinforcement) */
            add_scaled(mem[i].embedding, consensus, EMB_DIM, 0.05f);
            normalize(mem[i].embedding, EMB_DIM);
            mem[i].confidence = fminf(1.0f, mem[i].confidence + 0.02f);
        }
    }
}

/* --- Simple demonstration sequence: double recursive loop --- */
int main(int argc, char **argv) {
    MemoryBlock mem[MEMORY_SLOTS];
    int n_slots = 0;

    /* Initialize empty memory */
    for (int i = 0; i < MEMORY_SLOTS; ++i) mem[i].text[0] = '\0';

    /* Try loading prior state */
    int loaded = load_memory(MEMORY_FILE, mem, MEMORY_SLOTS);
    if (loaded > 0) {
        n_slots = loaded;
        printf("Loaded %d memory slots from %s\n", n_slots, MEMORY_FILE);
    } else {
        printf("No existing memory file; seeding with examples.\n");
        /* Seed with example assertions, including a contradictory or low-sim item */
        push_memory(mem, &n_slots, MEMORY_SLOTS, "Paris is the capital of France", 0.9f);
        push_memory(mem, &n_slots, MEMORY_SLOTS, "Paris is the largest city in France", 0.85f);
        push_memory(mem, &n_slots, MEMORY_SLOTS, "Lyon is the capital of France", 0.4f); /* contradictory */
        push_memory(mem, &n_slots, MEMORY_SLOTS, "France is in Europe", 0.95f);
    }

    /* Display initial memory */
    printf("Initial memory state:\n");
    for (int i = 0; i < n_slots; ++i) {
        printf(" %2d: conf=%.3f text=\"%s\"\n", i, mem[i].confidence, mem[i].text);
    }

    /* Double-pass (two recursive passes) */
    reconsider_pass(mem, n_slots, 1); /* pass 1: consensus/refine */
    reconsider_pass(mem, n_slots, 2); /* pass 2: contradiction detection & rewrite */

    /* Final state */
    printf("Final memory state after double recursive loop:\n");
    for (int i = 0; i < n_slots; ++i) {
        printf(" %2d: conf=%.3f text=\"%s\"\n", i, mem[i].confidence, mem[i].text);
    }

    /* Persist memory */
    if (save_memory(MEMORY_FILE, mem, n_slots)) {
        printf("Saved memory to %s\n", MEMORY_FILE);
    } else {
        fprintf(stderr, "Failed to save memory to %s\n", MEMORY_FILE);
    }

    return 0;
}
