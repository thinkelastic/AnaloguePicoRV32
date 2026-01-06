/*
 * PicoRV32 startup code
 * Sets up stack pointer and jumps to main
 */

.section .text.start
.global _start

_start:
    /* Set up stack pointer at end of RAM */
    /* PicoRV32 typically has RAM at 0x00000000 */
    /* Adjust STACK_TOP based on your memory map */
    lui sp, %hi(_stack_top)
    addi sp, sp, %lo(_stack_top)

    /* Clear BSS section */
    lui a0, %hi(_bss_start)
    addi a0, a0, %lo(_bss_start)
    lui a1, %hi(_bss_end)
    addi a1, a1, %lo(_bss_end)
clear_bss:
    bge a0, a1, done_bss
    sw zero, 0(a0)
    addi a0, a0, 4
    j clear_bss
done_bss:

    /* Jump to main */
    jal ra, main

    /* If main returns, loop forever */
halt:
    j halt
