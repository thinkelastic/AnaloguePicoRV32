//
// PicoRV32 CPU System
// - PicoRV32 RISC-V CPU
// - 4KB RAM for program/data (using block RAM)
// - Memory-mapped terminal at 0x20000000
//

`default_nettype none

module cpu_system (
    input wire clk,
    input wire reset_n,

    // Terminal memory interface (directly exposed to terminal module)
    output wire        term_mem_valid,
    output wire [31:0] term_mem_addr,
    output wire [31:0] term_mem_wdata,
    output wire [3:0]  term_mem_wstrb,
    input wire  [31:0] term_mem_rdata,
    input wire         term_mem_ready
);

// PicoRV32 signals
wire        mem_valid;
wire        mem_instr;
reg         mem_ready;
wire [31:0] mem_addr;
wire [31:0] mem_wdata;
wire [3:0]  mem_wstrb;
reg  [31:0] mem_rdata;

// Instantiate PicoRV32 CPU with minimal configuration
// This reduces logic utilization significantly
picorv32 #(
    .ENABLE_COUNTERS(0),
    .ENABLE_REGS_16_31(1),      // Enable all 32 registers (required by GCC)
    .ENABLE_REGS_DUALPORT(0),   // Use single-port register file
    .LATCHED_MEM_RDATA(0),
    .TWO_STAGE_SHIFT(1),        // Use smaller 2-stage shifter
    .TWO_CYCLE_COMPARE(1),      // Use 2-cycle compare
    .TWO_CYCLE_ALU(1),          // Use 2-cycle ALU
    .CATCH_MISALIGN(0),
    .CATCH_ILLINSN(0),
    .ENABLE_MUL(0),             // Disable multiplier
    .ENABLE_DIV(0),             // Disable divider
    .ENABLE_IRQ(0),             // Disable interrupts
    .ENABLE_IRQ_QREGS(0),
    .COMPRESSED_ISA(0)          // RV32I only, no compressed instructions
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
// 0x00000000 - 0x00000FFF : RAM (4KB)
// 0x20000000 - 0x200012BF : Terminal VRAM (4800 bytes, 80x60 characters)

// Decode memory regions
wire ram_select = (mem_addr[31:12] == 20'b0);           // 0x00000000-0x00000FFF (4KB)
wire term_select = (mem_addr[31:13] == 19'h10000);      // 0x20000000-0x20001FFF

// RAM using block RAM (4KB = 1024 x 32-bit words)
wire [31:0] ram_rdata;
wire [9:0] ram_addr = mem_addr[11:2];
wire ram_wren = mem_valid && !mem_ready && ram_select && |mem_wstrb;

altsyncram #(
    .operation_mode("SINGLE_PORT"),
    .width_a(32),
    .widthad_a(10),
    .numwords_a(1024),
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

// Memory access state machine
// BRAM has 1 cycle read latency
reg mem_pending;
reg ram_pending;
reg term_pending;

always @(posedge clk) begin
    mem_ready <= 0;

    if (!reset_n) begin
        mem_pending <= 0;
        ram_pending <= 0;
        term_pending <= 0;
    end else if (mem_valid && !mem_ready && !mem_pending) begin
        // Start of new memory access
        if (ram_select) begin
            // RAM access - need to wait 1 cycle for BRAM
            mem_pending <= 1;
            ram_pending <= 1;
            term_pending <= 0;
        end else if (term_select) begin
            // Terminal access - wait for terminal module
            mem_pending <= 1;
            ram_pending <= 0;
            term_pending <= 1;
        end else begin
            // Invalid address - return 0 immediately
            mem_ready <= 1;
            mem_rdata <= 32'h0;
        end
    end else if (mem_pending) begin
        if (ram_pending) begin
            // RAM data is now ready
            mem_ready <= 1;
            mem_rdata <= ram_rdata;
            mem_pending <= 0;
            ram_pending <= 0;
        end else if (term_pending && term_mem_ready) begin
            // Terminal data is ready
            mem_ready <= 1;
            mem_rdata <= term_mem_rdata;
            mem_pending <= 0;
            term_pending <= 0;
        end
    end
end

endmodule
