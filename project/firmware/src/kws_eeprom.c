#include <stdint.h>

#include "eeprom24c.h"
#include "kws_eeprom.h"

static int32_t read_i32_le(uint16_t addr)
{
    uint8_t b[4];
    if (!eeprom24c_read(addr, b, 4)) {
        return 0;
    }
    return (int32_t)((uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24));
}

uint8_t kws_eeprom_read_fc1_row(uint8_t row, int8_t *buf)
{
    uint16_t addr = (uint16_t)(KWS_EEPROM_FC1_W_OFFSET + (uint16_t)row * KWS_INPUT_SIZE);
    return eeprom24c_read(addr, (uint8_t *)buf, KWS_INPUT_SIZE);
}

int32_t kws_eeprom_read_fc1_bias(uint8_t row)
{
    uint16_t addr = (uint16_t)(KWS_EEPROM_FC1_B_OFFSET + (uint16_t)row * sizeof(int32_t));
    return read_i32_le(addr);
}

uint8_t kws_eeprom_read_fc2_row(uint8_t row, int8_t *buf)
{
    uint16_t addr = (uint16_t)(KWS_EEPROM_FC2_W_OFFSET + (uint16_t)row * KWS_HIDDEN_SIZE);
    return eeprom24c_read(addr, (uint8_t *)buf, KWS_HIDDEN_SIZE);
}

int32_t kws_eeprom_read_fc2_bias(uint8_t row)
{
    uint16_t addr = (uint16_t)(KWS_EEPROM_FC2_B_OFFSET + (uint16_t)row * sizeof(int32_t));
    return read_i32_le(addr);
}
