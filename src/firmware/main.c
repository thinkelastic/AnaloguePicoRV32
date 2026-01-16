/*
 * PicoRV32 Firmware for Analogue Pocket
 * Entry point - select test or llama mode
 */

#include "terminal.h"

/* External entry points */
extern void llama_main(void);
extern void memtest_main(void);

/* Set to 1 for memory test, 0 for llama */
#define RUN_MEMTEST 1

int main(void) {
    term_init();

    printf("PicoRV32 on Analogue Pocket\n");
    printf("===========================\n");
    printf("\n");

#if RUN_MEMTEST
    /* Run memory test suite */
    memtest_main();
#else
    /* Run Llama-2 inference */
    llama_main();
#endif

    /* Should not return, but if it does, idle */
    while (1) {
        /* Firmware idle loop */
    }

    return 0;
}
