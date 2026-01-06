# PicoRV32 RISC-V Core for Analogue Pocket

A demonstration core featuring a PicoRV32 RISC-V CPU running on the Analogue Pocket FPGA. The CPU executes custom firmware and displays output on a 40x30 character text terminal.

## Features

- **PicoRV32 RISC-V CPU** - Open source RV32I processor (ISC License)
- **4KB RAM** - Program and data storage
- **40x30 Text Terminal** - 1200 character display with 8x8 font
- **Printf Support** - Standard printf-style formatted output
- **320x240 @ 60Hz** - Native video output
- **Fully Open Source** - Redistributable under ISC license

## Quick Start

1. Copy the `release/Cores` and `release/Platforms` folders to your Analogue Pocket SD card root
2. Power on your Analogue Pocket
3. Navigate to Cores menu and select "Homebrew"
4. The core will display "Hello World!" on screen

## Project Structure

```
.
├── src/
│   ├── fpga/                     # FPGA source code
│   │   ├── apf/                  # Analogue Pocket Framework
│   │   ├── core/                 # Core implementation
│   │   │   ├── core_top.v        # Top-level module
│   │   │   ├── cpu_system.v      # PicoRV32 + RAM + peripherals
│   │   │   ├── text_terminal.v   # Text rendering
│   │   │   └── font_rom.v        # 8x8 character font
│   │   └── picorv32/             # PicoRV32 CPU core
│   │       └── picorv32.v
│   │
│   └── firmware/                 # RISC-V firmware
│       ├── main.c                # Main program
│       ├── terminal.c            # Terminal driver
│       ├── terminal.h            # Terminal API
│       ├── start.S               # Startup code
│       ├── linker.ld             # Linker script
│       └── Makefile              # Build system
│
├── release/                      # Ready for SD card
│   ├── Cores/ThinkElastic.Homebrew/
│   └── Platforms/
│
├── dist/                         # Distribution assets
│   └── platforms/
│
└── *.json                        # Core configuration files
```

## Technical Specifications

### Hardware

| Parameter | Value |
|-----------|-------|
| Target FPGA | Cyclone V 5CEBA4F23C8 |
| CPU Clock | 12.288 MHz |
| Video Output | 320x240 @ 60Hz |
| Program RAM | 4KB |
| VRAM | 1200 bytes (40x30 chars) |

### PicoRV32 Configuration

- RV32I instruction set (base integer)
- 32 registers (x0-x31)
- Single-port register file
- 2-stage shifter, 2-cycle ALU
- No hardware multiply/divide
- No interrupts

### Memory Map

| Address Range | Size | Description |
|--------------|------|-------------|
| `0x00000000 - 0x00000FFF` | 4KB | RAM (program + data) |
| `0x20000000 - 0x200004AF` | 1200B | VRAM (text terminal) |

## Firmware Development

### Prerequisites

Install the RISC-V toolchain:

```bash
# Arch Linux
sudo pacman -S riscv64-elf-gcc

# Ubuntu/Debian
sudo apt install gcc-riscv64-unknown-elf

# macOS (Homebrew)
brew install riscv64-elf-gcc
```

### Building Firmware

```bash
cd src/firmware
make clean all
make install    # Copy MIF to FPGA core directory
```

### Example Program

```c
#include "terminal.h"

int main(void) {
    term_init();

    printf("PicoRV32 on Analogue Pocket\n");
    printf("===========================\n");
    printf("\n");
    printf("Hello World!\n");

    while (1) {
        // Main loop
    }

    return 0;
}
```

### Terminal API

```c
void term_init(void);              // Initialize terminal
void term_clear(void);             // Clear screen
void term_setpos(int row, int col); // Set cursor position
void term_putchar(char c);         // Write single character
void term_puts(const char *s);     // Write string
void term_println(const char *s);  // Write string with newline
void term_printf(const char *fmt, ...); // Formatted output

// Printf supports: %d, %u, %x, %X, %s, %c, %%, width specifiers
#define printf term_printf         // Use printf() syntax
```

## Building the FPGA

### Prerequisites

- Intel Quartus Prime 25.1 or later
- Analogue Pocket development files

### Build Steps

1. Open `src/fpga/ap_core.qpf` in Quartus
2. Run compilation (Processing > Start Compilation)
3. The bitstream will be generated in `src/fpga/output_files/`

### Packaging for Analogue Pocket

After compilation, the release folder contains everything needed:
- `release/Cores/ThinkElastic.Homebrew/` - Core files
- `release/Platforms/` - Platform definition

Copy both folders to your SD card root.

## Key Design Decisions

### CPU Configuration

The PicoRV32 is configured with `ENABLE_REGS_16_31=1` to provide all 32 registers. This is required because GCC uses registers s2-s7 (x18-x23) for saved values, which are in the upper register set.

### Video Timing

Native 320x240 @ 60Hz output with 12.288 MHz pixel clock. The text terminal uses an 8x8 pixel font, giving 40 columns x 30 rows = 1200 characters.

### Memory Architecture

- Firmware RAM uses altsyncram with MIF initialization
- VRAM is memory-mapped and directly written by CPU
- Single-cycle memory access for both RAM and VRAM

## License

- **PicoRV32**: ISC License (Claire Wolf / YosysHQ)
- **Core Implementation**: ThinkElastic
- **Analogue Pocket Framework**: Provided by Analogue

## Resources

- [PicoRV32 GitHub](https://github.com/YosysHQ/picorv32)
- [Analogue Developer Portal](https://www.analogue.co/developer)
- [RISC-V Specifications](https://riscv.org/specifications/)
