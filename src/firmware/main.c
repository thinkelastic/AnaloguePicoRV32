/*
 * PicoRV32 Firmware for Analogue Pocket
 * Entry point for Llama-2 inference
 */

#include "terminal.h"

/* External entry point from llama_embedded.c */
extern void llama_main(void);

int main(void) {
    term_init();

    printf("PicoRV32 on Analogue Pocket\n");
    printf("===========================\n");
    printf("\n");

    /* Run Llama-2 inference */
    llama_main();

    /* Should not return, but if it does, idle */
    while (1) {
        /* Firmware idle loop */
    }

    return 0;
}
