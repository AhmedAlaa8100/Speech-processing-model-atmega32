#ifndef EEPROM24C_H
#define EEPROM24C_H

#include <stdint.h>

void eeprom24c_init(void);
uint8_t eeprom24c_read(uint16_t addr, uint8_t *buf, uint16_t len);

#endif
