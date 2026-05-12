#include <avr/io.h>
#include <stdint.h>

#include "sram.h"

#define SRAM_READ     0x03U
#define SRAM_WRITE    0x02U
#define SRAM_WRSR     0x01U
#define SRAM_MODE_SEQ 0x40U

#define cs_low()  (SRAM_CS_PORT &= (uint8_t) ~(1 << SRAM_CS_BIT))
#define cs_high() (SRAM_CS_PORT |= (uint8_t)(1 << SRAM_CS_BIT))

static uint8_t spi_xfer(uint8_t out)
{
    SPDR = out;
    while (!(SPSR & (1 << SPIF))) {
    }
    return SPDR;
}

static void sram_set_mode(uint8_t mode)
{
    cs_low();
    spi_xfer(SRAM_WRSR);
    spi_xfer(mode);
    cs_high();
}

void sram_init(void)
{
    SRAM_SPI_DDR |= (uint8_t)((1 << SRAM_SPI_MOSI) | (1 << SRAM_SPI_SCK) | (1 << SRAM_SPI_SS));
    SRAM_SPI_DDR &= (uint8_t) ~(1 << SRAM_SPI_MISO);

    SRAM_CS_DDR |= (uint8_t)(1 << SRAM_CS_BIT);
    cs_high();

    SPCR = (uint8_t)((1 << SPE) | (1 << MSTR));
    SPSR = 0;

    sram_set_mode(SRAM_MODE_SEQ);
}

void sram_begin_write(uint16_t addr)
{
    cs_low();
    spi_xfer(SRAM_WRITE);
    spi_xfer((uint8_t)(addr >> 8));
    spi_xfer((uint8_t)(addr & 0xFFU));
}

void sram_write_byte(uint8_t v)
{
    spi_xfer(v);
}

void sram_write_u16(int16_t v)
{
    uint16_t u = (uint16_t)v;
    sram_write_byte((uint8_t)(u >> 8));
    sram_write_byte((uint8_t)(u & 0xFFU));
}

void sram_end_write(void)
{
    cs_high();
}

void sram_begin_read(uint16_t addr)
{
    cs_low();
    spi_xfer(SRAM_READ);
    spi_xfer((uint8_t)(addr >> 8));
    spi_xfer((uint8_t)(addr & 0xFFU));
}

uint8_t sram_read_byte(void)
{
    return spi_xfer(0xFFU);
}

int16_t sram_read_u16(void)
{
    uint8_t hi = sram_read_byte();
    uint8_t lo = sram_read_byte();
    return (int16_t)(((uint16_t)hi << 8) | lo);
}

void sram_end_read(void)
{
    cs_high();
}

void sram_write_block(uint16_t addr, const uint8_t *buf, uint16_t len)
{
    sram_begin_write(addr);
    while (len--) {
        sram_write_byte(*buf++);
    }
    sram_end_write();
}

void sram_read_block(uint16_t addr, uint8_t *buf, uint16_t len)
{
    sram_begin_read(addr);
    while (len--) {
        *buf++ = sram_read_byte();
    }
    sram_end_read();
}
