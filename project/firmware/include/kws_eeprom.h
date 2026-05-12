#ifndef KWS_EEPROM_H
#define KWS_EEPROM_H

#include <stdint.h>

#include "config.h"
#include "kws_model.h"

#define KWS_EEPROM_FC1_W_OFFSET (KWS_EEPROM_BASE)
#define KWS_EEPROM_FC1_B_OFFSET ((uint16_t)(KWS_EEPROM_FC1_W_OFFSET + (KWS_HIDDEN_SIZE * KWS_INPUT_SIZE)))
#define KWS_EEPROM_FC2_W_OFFSET ((uint16_t)(KWS_EEPROM_FC1_B_OFFSET + (KWS_HIDDEN_SIZE * sizeof(int32_t))))
#define KWS_EEPROM_FC2_B_OFFSET ((uint16_t)(KWS_EEPROM_FC2_W_OFFSET + (KWS_NUM_CLASSES * KWS_HIDDEN_SIZE)))
#define KWS_EEPROM_TOTAL_BYTES ((uint16_t)(KWS_EEPROM_FC2_B_OFFSET + (KWS_NUM_CLASSES * sizeof(int32_t))))

uint8_t kws_eeprom_read_fc1_row(uint8_t row, int8_t *buf);
int32_t kws_eeprom_read_fc1_bias(uint8_t row);
uint8_t kws_eeprom_read_fc2_row(uint8_t row, int8_t *buf);
int32_t kws_eeprom_read_fc2_bias(uint8_t row);

#endif
