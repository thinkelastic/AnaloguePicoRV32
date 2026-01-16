/*
 * Full SDRAM Memory Test
 * Tests heap region in 1MB chunks with progress
 */

#include "libc/libc.h"
#include "terminal.h"
#include "dataslot.h"

#define printf term_printf

/* Memory addresses - all SDRAM now */
#define SDRAM_BASE      0x10000000
#define SDRAM_END       0x14000000  /* 64MB SDRAM */
#define HEAP_BASE       0x12100000  /* Heap starts after model data (~33MB in) */
#define HEAP_END        0x12200000  /* Test just 1MB first */
#define HEAP_SIZE       (HEAP_END - HEAP_BASE)  /* 1MB */
#define CHUNK_SIZE      (64 * 1024)  /* Test 64KB at a time */
#define CHUNK_WORDS     (CHUNK_SIZE / 4)

static uint32_t total_errors = 0;

/* Test a chunk and return error count */
static uint32_t test_chunk(volatile uint32_t* base, uint32_t count, uint32_t pattern) {
    uint32_t err = 0;

    /* Write */
    for (uint32_t i = 0; i < count; i++) {
        base[i] = pattern ^ i;
    }

    /* Verify */
    for (uint32_t i = 0; i < count; i++) {
        uint32_t expected = pattern ^ i;
        uint32_t got = base[i];
        if (got != expected) {
            if (err == 0) {
                printf("\n ERR@0x%08X w=%08X r=%08X",
                       (uint32_t)&base[i], expected, got);
            }
            err++;
        }
    }

    return err;
}

/* Speed test */
static void test_speed(volatile uint32_t* base, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) base[i] = i;

    uint32_t start = SYS_CYCLE_LO;
    volatile uint32_t sum = 0;
    for (uint32_t i = 0; i < count; i++) sum += base[i];
    uint32_t read_cycles = SYS_CYCLE_LO - start;

    start = SYS_CYCLE_LO;
    for (uint32_t i = 0; i < count; i++) base[i] = i;
    uint32_t write_cycles = SYS_CYCLE_LO - start;

    printf("Speed: R=%.1f W=%.1f cyc/word\n",
           (float)read_cycles / count,
           (float)write_cycles / count);
}

void memtest_main(void) {
    printf("=== Full SDRAM Test ===\n\n");

    while (!(SYS_STATUS & SYS_STATUS_SDRAM_READY));
    dataslot_wait_ready();
    printf("Hardware ready.\n");
    printf("Heap: 0x%08X-0x%08X (%dMB)\n\n",
           HEAP_BASE, HEAP_END, HEAP_SIZE / (1024 * 1024));

    /* Quick sanity test */
    volatile uint32_t* p = (volatile uint32_t*)HEAP_BASE;
    p[0] = 0xDEADBEEF;
    if (p[0] != 0xDEADBEEF) {
        printf("FAIL: Basic R/W broken!\n");
        while(1);
    }
    printf("Basic R/W: OK\n\n");

    /* Test full memory in chunks */
    uint32_t num_chunks = HEAP_SIZE / CHUNK_SIZE;
    uint32_t patterns[] = {0x5A5A5A5A, 0xFFFFFFFF, 0x00000000};

    for (int pass = 0; pass < 3; pass++) {
        uint32_t pattern = patterns[pass];
        uint32_t pass_errors = 0;

        printf("Pass %d (0x%08X): ", pass + 1, pattern);

        for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
            volatile uint32_t* base = (volatile uint32_t*)(HEAP_BASE + chunk * CHUNK_SIZE);
            uint32_t err = test_chunk(base, CHUNK_WORDS, pattern);
            pass_errors += err;

            /* Progress indicator */
            if ((chunk % 4) == 0) printf(".");
        }

        if (pass_errors == 0) {
            printf(" OK\n");
        } else {
            printf(" %d errs\n", pass_errors);
        }
        total_errors += pass_errors;
    }

    /* Speed test */
    printf("\n");
    test_speed((volatile uint32_t*)HEAP_BASE, 1024);

    /* Summary */
    printf("\n===================\n");
    if (total_errors == 0) {
        printf("ALL TESTS PASSED!\n");
        printf("%dMB verified OK\n", HEAP_SIZE / (1024 * 1024));
    } else {
        printf("ERRORS: %d\n", total_errors);
    }

    printf("\nDone.\n");
}
