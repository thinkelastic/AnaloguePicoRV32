//
// PicoRV32 CPU System
// - PicoRV32 RISC-V CPU with MUL/DIV extensions
// - 64KB RAM for program/data (using block RAM)
// - Memory-mapped terminal at 0x20000000
// - SDRAM access at 0x10000000 (64MB)
// - System registers at 0x40000000
//

`default_nettype none

module cpu_system (
    input wire clk,           // CPU clock (12.288 MHz)
    input wire clk_74a,       // Bridge/SDRAM clock (74.25 MHz)
    input wire reset_n,
    input wire dataslot_allcomplete,  // All data slots loaded by APF

    // Terminal memory interface
    output wire        term_mem_valid,
    output wire [31:0] term_mem_addr,
    output wire [31:0] term_mem_wdata,
    output wire [3:0]  term_mem_wstrb,
    input wire  [31:0] term_mem_rdata,
    input wire         term_mem_ready,

    // SDRAM word interface (directly to io_sdram via core_top)
    // These signals cross to clk_74a domain in core_top
    output reg         sdram_rd,
    output reg         sdram_wr,
    output reg  [23:0] sdram_addr,
    output reg  [31:0] sdram_wdata,
    input wire  [31:0] sdram_rdata,
    input wire         sdram_busy
);

// PicoRV32 signals
wire        mem_valid;
wire        mem_instr;
reg         mem_ready;
wire [31:0] mem_addr;
wire [31:0] mem_wdata;
wire [3:0]  mem_wstrb;
reg  [31:0] mem_rdata;

// Instantiate PicoRV32 CPU with MUL/DIV enabled for better performance
picorv32 #(
    .ENABLE_COUNTERS(1),        // Enable cycle counter for timing
    .ENABLE_COUNTERS64(1),      // 64-bit counter
    .ENABLE_REGS_16_31(1),      // Enable all 32 registers (required by GCC)
    .ENABLE_REGS_DUALPORT(1),   // Dual-port register file for better performance
    .LATCHED_MEM_RDATA(0),
    .TWO_STAGE_SHIFT(0),        // Single-cycle shift for performance
    .TWO_CYCLE_COMPARE(0),      // Single-cycle compare
    .TWO_CYCLE_ALU(0),          // Single-cycle ALU
    .CATCH_MISALIGN(0),
    .CATCH_ILLINSN(0),
    .ENABLE_MUL(1),             // Enable hardware multiplier
    .ENABLE_FAST_MUL(1),        // Fast multiplier
    .ENABLE_DIV(1),             // Enable hardware divider
    .ENABLE_IRQ(0),             // Disable interrupts (for now)
    .ENABLE_IRQ_QREGS(0),
    .COMPRESSED_ISA(0)          // RV32IM, no compressed instructions
) cpu (
    .clk(clk),
    .resetn(reset_n),
    .mem_valid(mem_valid),
    .mem_instr(mem_instr),
    .mem_ready(mem_ready),
    .mem_addr(mem_addr),
    .mem_wdata(mem_wdata),
    .mem_wstrb(mem_wstrb),
    .mem_rdata(mem_rdata)
);

// Memory map:
// 0x00000000 - 0x0000FFFF : RAM (64KB)
// 0x10000000 - 0x13FFFFFF : SDRAM (64MB)
// 0x20000000 - 0x20001FFF : Terminal VRAM
// 0x40000000 - 0x4000001F : System registers

// Decode memory regions
wire ram_select  = (mem_addr[31:16] == 16'b0);                    // 0x00000000-0x0000FFFF (64KB)
wire sdram_select = (mem_addr[31:26] == 6'b000100);               // 0x10000000-0x13FFFFFF (64MB)
wire term_select = (mem_addr[31:13] == 19'h10000);                // 0x20000000-0x20001FFF
wire sysreg_select = (mem_addr[31:5] == 27'h2000000);             // 0x40000000-0x4000001F

// RAM using block RAM (64KB = 16384 x 32-bit words)
wire [31:0] ram_rdata;
wire [13:0] ram_addr = mem_addr[15:2];
wire ram_wren = mem_valid && !mem_ready && ram_select && |mem_wstrb;

altsyncram #(
    .operation_mode("SINGLE_PORT"),
    .width_a(32),
    .widthad_a(14),              // 14 bits = 16384 words = 64KB
    .numwords_a(16384),
    .width_byteena_a(4),
    .lpm_type("altsyncram"),
    .outdata_reg_a("UNREGISTERED"),
    .init_file("core/firmware.mif"),
    .intended_device_family("Cyclone V"),
    .read_during_write_mode_port_a("NEW_DATA_NO_NBE_READ")
) ram (
    .clock0(clk),
    .address_a(ram_addr),
    .data_a(mem_wdata),
    .wren_a(ram_wren),
    .byteena_a(mem_wstrb),
    .q_a(ram_rdata),
    // Unused ports
    .aclr0(1'b0),
    .aclr1(1'b0),
    .address_b(1'b0),
    .addressstall_a(1'b0),
    .addressstall_b(1'b0),
    .byteena_b(1'b1),
    .clock1(1'b1),
    .clocken0(1'b1),
    .clocken1(1'b1),
    .clocken2(1'b1),
    .clocken3(1'b1),
    .data_b({32{1'b0}}),
    .eccstatus(),
    .q_b(),
    .rden_a(1'b1),
    .rden_b(1'b0),
    .wren_b(1'b0)
);

// Forward terminal requests to terminal module
assign term_mem_valid = mem_valid && term_select;
assign term_mem_addr = mem_addr;
assign term_mem_wdata = mem_wdata;
assign term_mem_wstrb = mem_wstrb;

// System registers (directly accessible)
// 0x00: SYS_STATUS   - Bit 0: always 1 (SDRAM ready), Bit 1: dataslot_allcomplete
// 0x04: SYS_CYCLE_LO - Cycle counter low
// 0x08: SYS_CYCLE_HI - Cycle counter high
reg [31:0] sysreg_rdata;
reg [63:0] cycle_counter;

// Synchronize dataslot_allcomplete from bridge clock domain (clk_74a) to CPU clock domain
reg [2:0] dataslot_allcomplete_sync;
always @(posedge clk) begin
    dataslot_allcomplete_sync <= {dataslot_allcomplete_sync[1:0], dataslot_allcomplete};
end
wire dataslot_allcomplete_s = dataslot_allcomplete_sync[2];

always @(posedge clk) begin
    if (!reset_n) begin
        cycle_counter <= 0;
    end else begin
        cycle_counter <= cycle_counter + 1;
    end
end

always @(*) begin
    case (mem_addr[4:2])
        3'b000: sysreg_rdata = {30'b0, dataslot_allcomplete_s, 1'b1};  // SYS_STATUS (SDRAM always ready)
        3'b001: sysreg_rdata = cycle_counter[31:0];   // SYS_CYCLE_LO
        3'b010: sysreg_rdata = cycle_counter[63:32];  // SYS_CYCLE_HI
        default: sysreg_rdata = 32'h0;
    endcase
end

// ============================================
// Memory access state machine
// ============================================
// Simple approach like the example - no busy checking, just fixed wait cycles
// 74.25 MHz / 12.288 MHz = ~6x clock ratio
// SDRAM operations take ~10-20 cycles at 74.25 MHz, so wait ~15 CPU cycles

reg mem_pending;
reg ram_pending;
reg term_pending;
reg sdram_pending;
reg sysreg_pending;
reg [4:0] sdram_wait;

always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        mem_ready <= 0;
        mem_rdata <= 0;
        mem_pending <= 0;
        ram_pending <= 0;
        term_pending <= 0;
        sdram_pending <= 0;
        sysreg_pending <= 0;
        sdram_rd <= 0;
        sdram_wr <= 0;
        sdram_addr <= 0;
        sdram_wdata <= 0;
        sdram_wait <= 0;
    end else begin
        mem_ready <= 0;
        sdram_rd <= 0;
        sdram_wr <= 0;

        if (sdram_wait > 0) begin
            sdram_wait <= sdram_wait - 1;
        end

        if (mem_valid && !mem_ready && !mem_pending) begin
            // Start of new memory access
            if (ram_select) begin
                mem_pending <= 1;
                ram_pending <= 1;
            end else if (sdram_select) begin
                // SDRAM access - issue request and start wait counter
                sdram_addr <= mem_addr[25:2];
                if (|mem_wstrb) begin
                    sdram_wr <= 1;
                    sdram_wdata <= mem_wdata;
                end else begin
                    sdram_rd <= 1;
                end
                sdram_wait <= 5'd15;  // Wait 15 CPU cycles for SDRAM
                mem_pending <= 1;
                sdram_pending <= 1;
            end else if (term_select) begin
                mem_pending <= 1;
                term_pending <= 1;
            end else if (sysreg_select) begin
                mem_pending <= 1;
                sysreg_pending <= 1;
            end else begin
                mem_ready <= 1;
                mem_rdata <= 32'h0;
            end
        end else if (mem_pending) begin
            if (ram_pending) begin
                mem_ready <= 1;
                mem_rdata <= ram_rdata;
                mem_pending <= 0;
                ram_pending <= 0;
            end else if (sdram_pending && sdram_wait == 0) begin
                // SDRAM wait complete - data should be ready
                mem_ready <= 1;
                mem_rdata <= sdram_rdata;
                mem_pending <= 0;
                sdram_pending <= 0;
            end else if (term_pending && term_mem_ready) begin
                mem_ready <= 1;
                mem_rdata <= term_mem_rdata;
                mem_pending <= 0;
                term_pending <= 0;
            end else if (sysreg_pending) begin
                mem_ready <= 1;
                mem_rdata <= sysreg_rdata;
                mem_pending <= 0;
                sysreg_pending <= 0;
            end
        end
    end
end

endmodule
