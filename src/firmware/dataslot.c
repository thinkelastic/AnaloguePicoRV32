/*
 * Data Slot driver implementation for Analogue Pocket
 *
 * SIMPLIFIED APPROACH:
 * APF automatically loads data slots to addresses specified in data.json.
 * The core just needs to wait for dataslot_allcomplete signal.
 *
 * Memory layout (defined in data.json):
 *   Slot 0 (Model):     0x10000000
 *   Slot 1 (Tokenizer): 0x12000000
 */

#include "dataslot.h"
#include "libc/libc.h"

/* Timeout for operations (in loop iterations) */
#define TIMEOUT_ITERATIONS  100000000

int dataslot_wait_ready(void) {
    /*
     * Wait for APF to finish loading all data slots.
     * APF loads data directly to SDRAM at addresses specified in data.json.
     * The dataslot_allcomplete flag is set when all slots are loaded.
     */
    int timeout = TIMEOUT_ITERATIONS;
    while (!(SYS_STATUS & SYS_STATUS_DATASLOT_COMPLETE)) {
        if (--timeout == 0) {
            return -1;  /* Timeout */
        }
    }
    return 0;
}

/* For compatibility - slot sizes aren't directly available with auto-loading */
int dataslot_get_size(uint16_t slot_id, uint32_t *size) {
    (void)slot_id;
    if (size == NULL) {
        return -1;
    }
    /* With auto-loading, we don't have size info readily available */
    /* Return -1 to indicate this function isn't supported in this mode */
    *size = 0;
    return -1;
}

/* For compatibility - not needed with auto-loading */
int dataslot_read(uint16_t slot_id, uint32_t offset, void *buffer, uint32_t length) {
    (void)slot_id;
    (void)offset;
    (void)buffer;
    (void)length;
    /* With auto-loading, data is already in SDRAM at fixed addresses */
    return -1;
}

/* For compatibility */
int32_t dataslot_load(uint16_t slot_id, void *dest) {
    (void)slot_id;
    (void)dest;
    /* With auto-loading, data is already loaded */
    return -1;
}

int32_t dataslot_load_to_addr(uint16_t slot_id, uint32_t sdram_addr) {
    (void)slot_id;
    (void)sdram_addr;
    /* With auto-loading, data is already loaded */
    return -1;
}
