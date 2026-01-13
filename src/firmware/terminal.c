/*
 * Simple terminal driver for PicoRV32
 * 40x30 character display at 0x20000000
 *
 * Note: Uses no division/modulo to work with RV32I base ISA
 */

#include "terminal.h"

// Terminal memory-mapped address
static volatile char *const terminal = (volatile char *)0x20000000;

// Current cursor position (0 to TERM_SIZE-1)
static int cursor_pos = 0;

// Get row from position (avoid division)
static int pos_to_row(int pos) {
    int row = 0;
    while (pos >= TERM_COLS) {
        pos -= TERM_COLS;
        row++;
    }
    return row;
}

// Get column from position (avoid modulo)
static int pos_to_col(int pos) {
    while (pos >= TERM_COLS) {
        pos -= TERM_COLS;
    }
    return pos;
}

// Scroll terminal up by one line
static void term_scroll(void) {
    // Copy each line to the line above
    for (int i = 0; i < TERM_SIZE - TERM_COLS; i++) {
        terminal[i] = terminal[i + TERM_COLS];
    }
    // Clear the bottom line
    for (int i = TERM_SIZE - TERM_COLS; i < TERM_SIZE; i++) {
        terminal[i] = ' ';
    }
}

void term_init(void) {
    cursor_pos = 0;
    term_clear();
}

void term_clear(void) {
    for (int i = 0; i < TERM_SIZE; i++) {
        terminal[i] = ' ';
    }
    cursor_pos = 0;
}

void term_setpos(int row, int col) {
    if (row < 0) row = 0;
    if (row >= TERM_ROWS) row = TERM_ROWS - 1;
    if (col < 0) col = 0;
    if (col >= TERM_COLS) col = TERM_COLS - 1;
    cursor_pos = row * TERM_COLS + col;
}

int term_getpos(void) {
    return cursor_pos;
}

void term_putchar(char c) {
    if (c == '\n') {
        // Move to start of next line
        int row = pos_to_row(cursor_pos);
        row++;
        if (row >= TERM_ROWS) {
            // Scroll up and stay on bottom line
            term_scroll();
            row = TERM_ROWS - 1;
        }
        cursor_pos = row * TERM_COLS;
    } else if (c == '\r') {
        // Move to start of current line
        int row = pos_to_row(cursor_pos);
        cursor_pos = row * TERM_COLS;
    } else if (c == '\t') {
        // Tab: move to next multiple of 4
        int col = pos_to_col(cursor_pos);
        col = (col + 4) & ~3;
        if (col >= TERM_COLS) {
            term_putchar('\n');
        } else {
            int row = pos_to_row(cursor_pos);
            cursor_pos = row * TERM_COLS + col;
        }
    } else if (c >= 32 && c < 127) {
        // Printable character
        terminal[cursor_pos] = c;
        cursor_pos++;
        if (cursor_pos >= TERM_SIZE) {
            // Scroll up and position at start of last line
            term_scroll();
            cursor_pos = (TERM_ROWS - 1) * TERM_COLS;
        }
    }
    // Ignore other control characters
}

void term_puts(const char *s) {
    while (*s) {
        term_putchar(*s++);
    }
}

void term_println(const char *s) {
    term_puts(s);
    term_putchar('\n');
}

void term_puthex(uint32_t val, int digits) {
    static const char hex[] = "0123456789ABCDEF";
    for (int i = digits - 1; i >= 0; i--) {
        term_putchar(hex[(val >> (i * 4)) & 0xF]);
    }
}

void term_putdec(int32_t val) {
    char buf[12];
    int i = 0;
    int neg = 0;
    uint32_t uval;

    if (val < 0) {
        neg = 1;
        uval = (uint32_t)(-val);
    } else {
        uval = (uint32_t)val;
    }

    // Handle zero
    if (uval == 0) {
        term_putchar('0');
        return;
    }

    // Build digits in reverse using subtraction
    while (uval > 0) {
        // Find uval % 10 and uval / 10 without division
        uint32_t q = 0;
        uint32_t r = uval;
        while (r >= 10) {
            r -= 10;
            q++;
        }
        buf[i++] = '0' + r;
        uval = q;
    }

    if (neg) {
        term_putchar('-');
    }

    // Output in correct order
    while (i > 0) {
        term_putchar(buf[--i]);
    }
}

static void print_unsigned(uint32_t val) {
    char buf[12];
    int i = 0;

    if (val == 0) {
        term_putchar('0');
        return;
    }

    while (val > 0) {
        uint32_t q = 0;
        uint32_t r = val;
        while (r >= 10) {
            r -= 10;
            q++;
        }
        buf[i++] = '0' + r;
        val = q;
    }

    while (i > 0) {
        term_putchar(buf[--i]);
    }
}

static void print_hex(uint32_t val, int uppercase) {
    const char *hex = uppercase ? "0123456789ABCDEF" : "0123456789abcdef";
    int started = 0;

    if (val == 0) {
        term_putchar('0');
        return;
    }

    for (int i = 7; i >= 0; i--) {
        int digit = (val >> (i * 4)) & 0xF;
        if (digit || started) {
            term_putchar(hex[digit]);
            started = 1;
        }
    }
}

void term_printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);

    while (*fmt) {
        if (*fmt == '%') {
            fmt++;
            switch (*fmt) {
                case 'd':
                case 'i':
                    term_putdec(va_arg(args, int32_t));
                    break;
                case 'u':
                    print_unsigned(va_arg(args, uint32_t));
                    break;
                case 'x':
                    print_hex(va_arg(args, uint32_t), 0);
                    break;
                case 'X':
                    print_hex(va_arg(args, uint32_t), 1);
                    break;
                case 's': {
                    const char *s = va_arg(args, const char *);
                    if (s) term_puts(s);
                    else term_puts("(null)");
                    break;
                }
                case 'c':
                    term_putchar((char)va_arg(args, int));
                    break;
                case '%':
                    term_putchar('%');
                    break;
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9': {
                    // Simple width specifier for hex (e.g., %08X)
                    int width = 0;
                    while (*fmt >= '0' && *fmt <= '9') {
                        width = width * 10 + (*fmt - '0');
                        fmt++;
                    }
                    if (*fmt == 'x' || *fmt == 'X') {
                        uint32_t val = va_arg(args, uint32_t);
                        const char *hex = (*fmt == 'X') ? "0123456789ABCDEF" : "0123456789abcdef";
                        if (width > 8) width = 8;
                        for (int i = width - 1; i >= 0; i--) {
                            term_putchar(hex[(val >> (i * 4)) & 0xF]);
                        }
                    }
                    break;
                }
                default:
                    term_putchar('%');
                    term_putchar(*fmt);
                    break;
            }
        } else {
            term_putchar(*fmt);
        }
        fmt++;
    }

    va_end(args);
}
