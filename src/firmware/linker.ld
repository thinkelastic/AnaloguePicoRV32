/*
 * Linker script for PicoRV32 on Analogue Pocket
 *
 * Memory layout:
 * - RAM:      0x00000000 (64KB) - Code, data, and stack (initialized from firmware.mif)
 * - SDRAM:    0x10000000 (64MB) - Model weights, tokenizer, and heap
 * - Terminal: 0x20000000 (8KB)  - Character VRAM
 * - DataSlot: 0x30000000 (64B)  - Data slot loader registers
 * - SysRegs:  0x40000000 (32B)  - System control registers
 */

ENTRY(_start)

MEMORY {
    RAM   (rwx) : ORIGIN = 0x00000000, LENGTH = 64K
    SDRAM (rwx) : ORIGIN = 0x10000000, LENGTH = 64M
}

/* SDRAM layout symbols */
__sdram_base = 0x10000000;
__sdram_size = 64M;

SECTIONS {
    /* Code section - starts at 0 */
    .text : {
        KEEP(*(.text.start))   /* Startup code first */
        *(.text*)              /* All other code */
        *(.rodata*)            /* Read-only data */
        . = ALIGN(4);
        __text_end = .;
    } > RAM

    /* Initialized data */
    .data : {
        __data_start = .;
        *(.data*)
        *(.sdata*)
        . = ALIGN(4);
        __data_end = .;
    } > RAM

    /* Uninitialized data (BSS) */
    .bss : {
        . = ALIGN(4);
        __bss_start = .;
        *(.bss*)
        *(.sbss*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > RAM

    /* End of used RAM - for heap/stack calculations */
    __ram_end = .;

    /* Stack at end of RAM (grows downward) */
    __stack_top = ORIGIN(RAM) + LENGTH(RAM);

    /* SDRAM heap section (not actually placed, just defines symbols) */
    .heap (NOLOAD) : {
        __heap_start = .;
        . = ORIGIN(SDRAM) + LENGTH(SDRAM);
        __heap_end = .;
    } > SDRAM

    /* Provide symbols for C code */
    PROVIDE(_stack_top = __stack_top);
    PROVIDE(_bss_start = __bss_start);
    PROVIDE(_bss_end = __bss_end);
    PROVIDE(_heap_start = __sdram_base);
    PROVIDE(_heap_end = __sdram_base + __sdram_size);
}
