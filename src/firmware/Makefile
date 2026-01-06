# PicoRV32 Firmware Makefile for Analogue Pocket

# Toolchain
CROSS = riscv64-elf-
CC = $(CROSS)gcc
AS = $(CROSS)gcc
LD = $(CROSS)gcc
OBJCOPY = $(CROSS)objcopy
OBJDUMP = $(CROSS)objdump
SIZE = $(CROSS)size

# Target
TARGET = firmware

# Directories
FPGA_CORE_DIR = ../fpga/core

# Source files
SRCS_C = main.c terminal.c
SRCS_S = start.S
OBJS = $(SRCS_C:.c=.o) $(SRCS_S:.S=.o)

# Compiler flags for RV32I (base integer only, no multiply/divide)
ARCH = rv32i
ABI = ilp32
CFLAGS = -march=$(ARCH) -mabi=$(ABI) -Os -Wall -Wextra
CFLAGS += -ffreestanding -nostdlib -nostartfiles
CFLAGS += -ffunction-sections -fdata-sections

# Assembler flags
ASFLAGS = -march=$(ARCH) -mabi=$(ABI)

# Linker flags
LDFLAGS = -march=$(ARCH) -mabi=$(ABI)
LDFLAGS += -T linker.ld -nostdlib -nostartfiles
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -Wl,-Map=$(TARGET).map

# Default target
all: $(TARGET).bin $(TARGET).mif $(TARGET).lst
	$(SIZE) $(TARGET).elf

# Link
$(TARGET).elf: $(OBJS) linker.ld
	$(LD) $(LDFLAGS) -o $@ $(OBJS)

# Binary output
$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@

# Quartus MIF output (Memory Initialization File)
# Converts binary to MIF format for altsyncram initialization
$(TARGET).mif: $(TARGET).bin
	@echo "-- Firmware RAM initialization - 1024 x 32-bit words (4KB)" > $@
	@echo "-- Auto-generated from $(TARGET).bin" >> $@
	@echo "" >> $@
	@echo "WIDTH=32;" >> $@
	@echo "DEPTH=1024;" >> $@
	@echo "" >> $@
	@echo "ADDRESS_RADIX=DEC;" >> $@
	@echo "DATA_RADIX=HEX;" >> $@
	@echo "" >> $@
	@echo "CONTENT BEGIN" >> $@
	@hexdump -v -e '1/4 "%08X\n"' $< | awk '{printf "%d : %s;\n", NR-1, $$1}' >> $@
	@LAST=$$(hexdump -v -e '1/4 "%08X\n"' $< | wc -l); \
	if [ $$LAST -lt 1024 ]; then \
		echo "[$$LAST..1023] : 00000013;" >> $@; \
	fi
	@echo "END;" >> $@
	@echo "Generated $@ with $$(hexdump -v -e '1/4 "%08X\n"' $< | wc -l) words of firmware"

# Disassembly listing
$(TARGET).lst: $(TARGET).elf
	$(OBJDUMP) -d -S $< > $@

# Compile C sources
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Assemble assembly sources
%.o: %.S
	$(AS) $(ASFLAGS) -c -o $@ $<

# Install MIF to FPGA core directory
install: $(TARGET).bin $(TARGET).mif
	cp $(TARGET).mif $(FPGA_CORE_DIR)/firmware.mif
	@echo "MIF installed to $(FPGA_CORE_DIR)/firmware.mif"

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET).elf $(TARGET).bin $(TARGET).mif $(TARGET).map $(TARGET).lst

# Rebuild everything
rebuild: clean all

.PHONY: all clean rebuild install
