/*
 * File I/O emulation for PicoRV32
 * Uses data slots for loading files into SDRAM
 */

#include "libc.h"
#include "../dataslot.h"

/* Standard file descriptors (unused but defined for compatibility) */
static FILE stdin_file = {0, 0, 0, 0, NULL};
static FILE stdout_file = {0, 0, 0, 0, NULL};
static FILE stderr_file = {0, 0, 0, 0, NULL};

FILE *stdin = &stdin_file;
FILE *stdout = &stdout_file;
FILE *stderr = &stderr_file;

/* File table for open files */
#define MAX_OPEN_FILES 4
static FILE file_table[MAX_OPEN_FILES];
static int file_table_used[MAX_OPEN_FILES] = {0};

/* Find a free file slot */
static FILE *alloc_file(void) {
    for (int i = 0; i < MAX_OPEN_FILES; i++) {
        if (!file_table_used[i]) {
            file_table_used[i] = 1;
            memset(&file_table[i], 0, sizeof(FILE));
            return &file_table[i];
        }
    }
    return NULL;
}

/* Free a file slot */
static void free_file(FILE *f) {
    for (int i = 0; i < MAX_OPEN_FILES; i++) {
        if (&file_table[i] == f) {
            file_table_used[i] = 0;
            return;
        }
    }
}

/* Map filename to data slot ID */
static int filename_to_slot(const char *pathname) {
    /* Check for known filenames */
    if (strcmp(pathname, "model.bin") == 0 ||
        strrchr(pathname, '/') != NULL &&
        strcmp(strrchr(pathname, '/') + 1, "model.bin") == 0) {
        return 0;
    }
    if (strcmp(pathname, "tokenizer.bin") == 0 ||
        strrchr(pathname, '/') != NULL &&
        strcmp(strrchr(pathname, '/') + 1, "tokenizer.bin") == 0) {
        return 1;
    }

    /* Unknown file */
    return -1;
}

/* ============================================
 * High-level file operations
 * ============================================ */

FILE *fopen(const char *pathname, const char *mode) {
    (void)mode;  /* Only read mode is supported */

    int slot_id = filename_to_slot(pathname);
    if (slot_id < 0) {
        return NULL;
    }

    FILE *f = alloc_file();
    if (f == NULL) {
        return NULL;
    }

    f->slot_id = slot_id;
    f->offset = 0;
    f->flags = 0;

    /* Get slot size */
    if (dataslot_get_size(slot_id, &f->size) != 0) {
        free_file(f);
        return NULL;
    }

    /* Data will be loaded into SDRAM on first read or via mmap */
    f->data = NULL;

    return f;
}

int fclose(FILE *stream) {
    if (stream == NULL) {
        return -1;
    }

    /* Don't free SDRAM data here - mmap/munmap handles that */
    free_file(stream);
    return 0;
}

size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    if (stream == NULL || ptr == NULL || size == 0 || nmemb == 0) {
        return 0;
    }

    size_t total_bytes = size * nmemb;
    size_t available = stream->size - stream->offset;

    if (total_bytes > available) {
        total_bytes = available;
        nmemb = total_bytes / size;
        total_bytes = nmemb * size;  /* Round down to whole elements */
    }

    if (total_bytes == 0) {
        return 0;
    }

    /* If data is loaded in memory (via mmap), copy directly from it */
    if (stream->data != NULL) {
        memcpy(ptr, (uint8_t *)stream->data + stream->offset, total_bytes);
        stream->offset += total_bytes;
        return nmemb;
    }

    /* Otherwise, load directly from data slot to provided buffer */
    /* This is slower but works for non-mmap usage */
    if (dataslot_read(stream->slot_id, stream->offset, ptr, total_bytes) != 0) {
        return 0;
    }

    stream->offset += total_bytes;
    return nmemb;
}

size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream) {
    (void)ptr;
    (void)size;
    (void)nmemb;
    (void)stream;
    /* Write not supported for data slots */
    return 0;
}

int fseek(FILE *stream, long offset, int whence) {
    if (stream == NULL) {
        return -1;
    }

    long new_offset;

    switch (whence) {
        case SEEK_SET:
            new_offset = offset;
            break;
        case SEEK_CUR:
            new_offset = (long)stream->offset + offset;
            break;
        case SEEK_END:
            new_offset = (long)stream->size + offset;
            break;
        default:
            return -1;
    }

    if (new_offset < 0 || (size_t)new_offset > stream->size) {
        return -1;
    }

    stream->offset = (uint32_t)new_offset;
    return 0;
}

long ftell(FILE *stream) {
    if (stream == NULL) {
        return -1;
    }
    return (long)stream->offset;
}

void rewind(FILE *stream) {
    if (stream != NULL) {
        stream->offset = 0;
    }
}

int fflush(FILE *stream) {
    (void)stream;
    return 0;  /* Nothing to flush for read-only files */
}

int feof(FILE *stream) {
    if (stream == NULL) {
        return 1;
    }
    return stream->offset >= stream->size;
}

int ferror(FILE *stream) {
    (void)stream;
    return 0;  /* No error tracking implemented */
}

/* ============================================
 * Formatted I/O (minimal implementation)
 * ============================================ */

/* Forward declaration from terminal */
extern void term_printf(const char *fmt, ...);

int fprintf(FILE *stream, const char *format, ...) {
    (void)stream;
    /* Just print to terminal */
    va_list args;
    va_start(args, format);
    /* Note: This is a simplified implementation */
    /* For full fprintf, would need vfprintf implementation */
    term_printf(format);
    va_end(args);
    return 0;
}

int sprintf(char *str, const char *format, ...) {
    /* Simplified sprintf - only handles basic cases */
    va_list args;
    va_start(args, format);

    char *out = str;
    const char *p = format;

    while (*p) {
        if (*p == '%') {
            p++;
            switch (*p) {
                case 'd':
                case 'i': {
                    int val = va_arg(args, int);
                    if (val < 0) {
                        *out++ = '-';
                        val = -val;
                    }
                    /* Convert to string (reversed) */
                    char buf[12];
                    int i = 0;
                    do {
                        buf[i++] = '0' + (val % 10);
                        val /= 10;
                    } while (val > 0);
                    while (i > 0) {
                        *out++ = buf[--i];
                    }
                    break;
                }
                case 'u': {
                    unsigned int val = va_arg(args, unsigned int);
                    char buf[12];
                    int i = 0;
                    do {
                        buf[i++] = '0' + (val % 10);
                        val /= 10;
                    } while (val > 0);
                    while (i > 0) {
                        *out++ = buf[--i];
                    }
                    break;
                }
                case 's': {
                    const char *s = va_arg(args, const char *);
                    if (s == NULL) s = "(null)";
                    while (*s) {
                        *out++ = *s++;
                    }
                    break;
                }
                case 'c': {
                    char c = (char)va_arg(args, int);
                    *out++ = c;
                    break;
                }
                case '%':
                    *out++ = '%';
                    break;
                default:
                    *out++ = '%';
                    *out++ = *p;
                    break;
            }
        } else {
            *out++ = *p;
        }
        p++;
    }

    *out = '\0';
    va_end(args);
    return out - str;
}

int snprintf(char *str, size_t size, const char *format, ...) {
    /* Just use sprintf and hope size is enough */
    /* A proper implementation would respect size */
    (void)size;
    va_list args;
    va_start(args, format);
    /* Simplified - doesn't actually limit to size */
    int result = sprintf(str, format);
    va_end(args);
    return result;
}

int sscanf(const char *str, const char *format, ...) {
    va_list args;
    va_start(args, format);

    int count = 0;
    const char *s = str;
    const char *f = format;

    while (*f && *s) {
        if (*f == '%') {
            f++;
            switch (*f) {
                case 'd':
                case 'i': {
                    int *ptr = va_arg(args, int *);
                    int sign = 1;
                    int val = 0;

                    /* Skip whitespace */
                    while (isspace(*s)) s++;

                    if (*s == '-') {
                        sign = -1;
                        s++;
                    } else if (*s == '+') {
                        s++;
                    }

                    if (!isdigit(*s)) break;

                    while (isdigit(*s)) {
                        val = val * 10 + (*s - '0');
                        s++;
                    }

                    *ptr = val * sign;
                    count++;
                    break;
                }
                case 'f': {
                    float *ptr = va_arg(args, float *);

                    /* Skip whitespace */
                    while (isspace(*s)) s++;

                    /* Parse float using atof logic */
                    const char *start = s;
                    int sign = 1;
                    float val = 0.0f;
                    float frac = 0.0f;
                    float div = 1.0f;
                    int in_frac = 0;

                    if (*s == '-') { sign = -1; s++; }
                    else if (*s == '+') { s++; }

                    while (*s && (isdigit(*s) || *s == '.')) {
                        if (*s == '.') {
                            if (in_frac) break;
                            in_frac = 1;
                        } else if (in_frac) {
                            div *= 10.0f;
                            frac += (*s - '0') / div;
                        } else {
                            val = val * 10.0f + (*s - '0');
                        }
                        s++;
                    }

                    if (s == start) break;

                    *ptr = (val + frac) * sign;
                    count++;
                    break;
                }
                case 'x':
                case 'X': {
                    unsigned int *ptr = va_arg(args, unsigned int *);
                    unsigned int val = 0;

                    while (isspace(*s)) s++;

                    /* Skip 0x prefix if present */
                    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
                        s += 2;
                    }

                    while (*s) {
                        if (isdigit(*s)) {
                            val = val * 16 + (*s - '0');
                        } else if (*s >= 'a' && *s <= 'f') {
                            val = val * 16 + (*s - 'a' + 10);
                        } else if (*s >= 'A' && *s <= 'F') {
                            val = val * 16 + (*s - 'A' + 10);
                        } else {
                            break;
                        }
                        s++;
                    }

                    *ptr = val;
                    count++;
                    break;
                }
                default:
                    break;
            }
            f++;
        } else if (isspace(*f)) {
            /* Skip whitespace in both format and input */
            while (isspace(*f)) f++;
            while (isspace(*s)) s++;
        } else {
            /* Literal match */
            if (*f != *s) break;
            f++;
            s++;
        }
    }

    va_end(args);
    return count;
}

/* ============================================
 * POSIX-style file operations
 * ============================================ */

/* Use negative slot IDs as file descriptors */
#define FD_TO_SLOT(fd) (-(fd) - 1)
#define SLOT_TO_FD(slot) (-(slot) - 1)

static uint32_t fd_offset[16] = {0};
static uint32_t fd_size[16] = {0};
static int fd_used[16] = {0};

int open(const char *pathname, int flags) {
    (void)flags;  /* Only O_RDONLY supported */

    int slot_id = filename_to_slot(pathname);
    if (slot_id < 0 || slot_id >= 16) {
        return -1;
    }

    if (fd_used[slot_id]) {
        return -1;  /* Already open */
    }

    /* Get size */
    if (dataslot_get_size(slot_id, &fd_size[slot_id]) != 0) {
        return -1;
    }

    fd_offset[slot_id] = 0;
    fd_used[slot_id] = 1;

    return SLOT_TO_FD(slot_id);
}

int close(int fd) {
    int slot_id = FD_TO_SLOT(fd);
    if (slot_id < 0 || slot_id >= 16 || !fd_used[slot_id]) {
        return -1;
    }

    fd_used[slot_id] = 0;
    return 0;
}

ssize_t read(int fd, void *buf, size_t count) {
    int slot_id = FD_TO_SLOT(fd);
    if (slot_id < 0 || slot_id >= 16 || !fd_used[slot_id]) {
        return -1;
    }

    uint32_t available = fd_size[slot_id] - fd_offset[slot_id];
    if (count > available) {
        count = available;
    }

    if (count == 0) {
        return 0;
    }

    if (dataslot_read(slot_id, fd_offset[slot_id], buf, count) != 0) {
        return -1;
    }

    fd_offset[slot_id] += count;
    return count;
}

off_t lseek(int fd, off_t offset, int whence) {
    int slot_id = FD_TO_SLOT(fd);
    if (slot_id < 0 || slot_id >= 16 || !fd_used[slot_id]) {
        return -1;
    }

    off_t new_offset;
    switch (whence) {
        case SEEK_SET:
            new_offset = offset;
            break;
        case SEEK_CUR:
            new_offset = fd_offset[slot_id] + offset;
            break;
        case SEEK_END:
            new_offset = fd_size[slot_id] + offset;
            break;
        default:
            return -1;
    }

    if (new_offset < 0) {
        return -1;
    }

    fd_offset[slot_id] = new_offset;
    return new_offset;
}

/* ============================================
 * mmap emulation
 * ============================================ */

/* Static buffer for mmap'd data - we allocate from SDRAM heap */
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    (void)addr;
    (void)prot;
    (void)flags;

    int slot_id = FD_TO_SLOT(fd);
    if (slot_id < 0 || slot_id >= 16 || !fd_used[slot_id]) {
        return MAP_FAILED;
    }

    /* Allocate memory for the mapped region */
    void *ptr = malloc(length);
    if (ptr == NULL) {
        return MAP_FAILED;
    }

    /* Load data from data slot */
    if (dataslot_read(slot_id, offset, ptr, length) != 0) {
        free(ptr);
        return MAP_FAILED;
    }

    return ptr;
}

int munmap(void *addr, size_t length) {
    (void)length;
    if (addr != NULL && addr != MAP_FAILED) {
        free(addr);
    }
    return 0;
}
