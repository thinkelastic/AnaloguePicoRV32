/*
 * Linker script for PicoRV32 on Analogue Pocket
 *
 * Memory layout:
 * - RAM:      0x00000000 (4KB) - Code and data (initialized from firmware.hex)
 * - Terminal: 0x20000000 (4800 bytes) - 80x60 character VRAM
 */

ENTRY(_start)

MEMORY {
    RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 4K
}

SECTIONS {
    /* Code section */
    .text : {
        KEEP(*(.text.start))   /* Startup code first */
        *(.text*)              /* All other code */
        *(.rodata*)            /* Read-only data */
        . = ALIGN(4);
    } > RAM

    /* Initialized data */
    .data : {
        _data_start = .;
        *(.data*)
        . = ALIGN(4);
        _data_end = .;
    } > RAM

    /* Uninitialized data (BSS) */
    .bss : {
        . = ALIGN(4);
        _bss_start = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _bss_end = .;
    } > RAM

    /* Stack at end of RAM */
    _stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
