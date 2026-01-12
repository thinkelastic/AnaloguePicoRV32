/*
 * Data Slot driver for Analogue Pocket
 * Provides interface to load data from SD card via APF data slots
 */

#ifndef DATASLOT_H
#define DATASLOT_H

#include <stdint.h>
#include <stddef.h>

/* Data slot loader register base address */
#define DATASLOT_BASE   0x30000000

/* Data slot loader registers */
#define DATASLOT_CTRL       (*(volatile uint32_t*)(DATASLOT_BASE + 0x00))
#define DATASLOT_SLOT_ID    (*(volatile uint32_t*)(DATASLOT_BASE + 0x04))
#define DATASLOT_SLOT_OFF   (*(volatile uint32_t*)(DATASLOT_BASE + 0x08))
#define DATASLOT_DEST_ADDR  (*(volatile uint32_t*)(DATASLOT_BASE + 0x0C))
#define DATASLOT_LENGTH     (*(volatile uint32_t*)(DATASLOT_BASE + 0x10))
#define DATASLOT_SIZE_LO    (*(volatile uint32_t*)(DATASLOT_BASE + 0x14))
#define DATASLOT_SIZE_HI    (*(volatile uint32_t*)(DATASLOT_BASE + 0x18))
#define DATASLOT_STATUS     (*(volatile uint32_t*)(DATASLOT_BASE + 0x1C))

/* Control register bits */
#define DATASLOT_CTRL_START     0x01    /* Start transfer */
#define DATASLOT_CTRL_QUERY     0x02    /* Query slot size */

/* Status register bits */
#define DATASLOT_STATUS_BUSY    0x01
#define DATASLOT_STATUS_DONE    0x02
#define DATASLOT_STATUS_ERROR   0x04
#define DATASLOT_STATUS_ERR_MASK 0xF0   /* Error code in bits 7:4 */

/* Data slot IDs */
#define SLOT_MODEL      0
#define SLOT_TOKENIZER  1

/**
 * Wait for data slots to be ready
 * Call this at startup to ensure all data slots are loaded
 * Returns 0 on success, -1 on timeout
 */
int dataslot_wait_ready(void);

/**
 * Get the size of a data slot
 * @param slot_id Data slot ID (0 = model, 1 = tokenizer)
 * @param size Pointer to store the size
 * @return 0 on success, -1 on error
 */
int dataslot_get_size(uint16_t slot_id, uint32_t *size);

/**
 * Read data from a data slot into a buffer
 * @param slot_id Data slot ID
 * @param offset Offset within the slot
 * @param buffer Destination buffer (must be in SDRAM)
 * @param length Number of bytes to read
 * @return 0 on success, -1 on error
 */
int dataslot_read(uint16_t slot_id, uint32_t offset, void *buffer, uint32_t length);

/**
 * Load entire data slot into SDRAM
 * @param slot_id Data slot ID
 * @param dest SDRAM destination address
 * @return Number of bytes loaded, or -1 on error
 */
int32_t dataslot_load(uint16_t slot_id, void *dest);

/**
 * Load entire data slot into SDRAM at specified address
 * @param slot_id Data slot ID
 * @param sdram_addr SDRAM address (must be >= SDRAM_BASE)
 * @return Number of bytes loaded, or -1 on error
 */
int32_t dataslot_load_to_addr(uint16_t slot_id, uint32_t sdram_addr);

#endif /* DATASLOT_H */
