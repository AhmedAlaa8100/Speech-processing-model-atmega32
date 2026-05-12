#ifndef SRAM_H
#define SRAM_H

#include <stdint.h>

/*
 * 23K256 / SPI SRAM wiring (edit to match your PCB).
 * ATmega32 hardware SPI: MOSI PB5, MISO PB6, SCK PB7.
 * CS on PB2 (PB0 is the record button in main.c).
 */
#define SRAM_SPI_DDR  DDRB
#define SRAM_SPI_MOSI PB5
#define SRAM_SPI_MISO PB6
#define SRAM_SPI_SCK  PB7
#define SRAM_SPI_SS   PB4

#define SRAM_CS_PORT PORTB
#define SRAM_CS_DDR  DDRB
#define SRAM_CS_BIT  PB2

void sram_init(void);

void sram_begin_write(uint16_t addr);
void sram_write_byte(uint8_t v);
void sram_write_u16(int16_t v);
void sram_end_write(void);

void sram_begin_read(uint16_t addr);
uint8_t sram_read_byte(void);
int16_t sram_read_u16(void);
void sram_end_read(void);

void sram_write_block(uint16_t addr, const uint8_t *buf, uint16_t len);
void sram_read_block(uint16_t addr, uint8_t *buf, uint16_t len);

#endif
