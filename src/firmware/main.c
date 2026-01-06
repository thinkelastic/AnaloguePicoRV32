/*
 * PicoRV32 Firmware for Analogue Pocket
 */

#include "terminal.h"

int main(void) {
    term_init();

    printf("PicoRV32 on Analogue Pocket\n");
    printf("===========================\n");
    printf("\n");
    printf("Hello World!\n");

    /* Main loop */
    while (1) {
        /* Firmware idle loop */
    }

    return 0;
}
