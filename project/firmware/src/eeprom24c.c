#ifndef F_CPU
#define F_CPU 11059200UL
#endif

#include <avr/io.h>
#include <stdint.h>
#include <util/delay.h>

#include "config.h"
#include "eeprom24c.h"

#define SDA_BIT EEPROM_I2C_SDA_BIT
#define SCL_BIT EEPROM_I2C_SCL_BIT

#define I2C_PORT PORTC
#define I2C_PIN  PINC
#define I2C_DDR  DDRC

static void i2c_delay(void)
{
    _delay_us(EEPROM_I2C_DELAY_US);
}

static void sda_high(void)
{
    I2C_DDR &= (uint8_t)~(1 << SDA_BIT);
    I2C_PORT |= (uint8_t)(1 << SDA_BIT);
}

static void sda_low(void)
{
    I2C_DDR |= (uint8_t)(1 << SDA_BIT);
    I2C_PORT &= (uint8_t)~(1 << SDA_BIT);
}

static void scl_high(void)
{
    I2C_DDR &= (uint8_t)~(1 << SCL_BIT);
    I2C_PORT |= (uint8_t)(1 << SCL_BIT);
}

static void scl_low(void)
{
    I2C_DDR |= (uint8_t)(1 << SCL_BIT);
    I2C_PORT &= (uint8_t)~(1 << SCL_BIT);
}

static void i2c_start(void)
{
    sda_high();
    scl_high();
    i2c_delay();
    sda_low();
    i2c_delay();
    scl_low();
}

static void i2c_stop(void)
{
    sda_low();
    i2c_delay();
    scl_high();
    i2c_delay();
    sda_high();
    i2c_delay();
}

static uint8_t i2c_write_byte(uint8_t data)
{
    for (uint8_t mask = 0x80U; mask != 0; mask >>= 1) {
        if (data & mask) {
            sda_high();
        } else {
            sda_low();
        }
        i2c_delay();
        scl_high();
        i2c_delay();
        scl_low();
    }

    sda_high();
    i2c_delay();
    scl_high();
    i2c_delay();
    uint8_t ack = (uint8_t)((I2C_PIN & (1 << SDA_BIT)) == 0U);
    scl_low();
    i2c_delay();
    return ack;
}

static uint8_t i2c_read_byte(uint8_t ack)
{
    uint8_t data = 0;
    sda_high();
    for (uint8_t i = 0; i < 8U; i++) {
        data <<= 1;
        scl_high();
        i2c_delay();
        if (I2C_PIN & (1 << SDA_BIT)) {
            data |= 1U;
        }
        scl_low();
        i2c_delay();
    }

    if (ack) {
        sda_low();
    } else {
        sda_high();
    }
    i2c_delay();
    scl_high();
    i2c_delay();
    scl_low();
    sda_high();
    i2c_delay();
    return data;
}

void eeprom24c_init(void)
{
    sda_high();
    scl_high();
}

uint8_t eeprom24c_read(uint16_t addr, uint8_t *buf, uint16_t len)
{
    uint8_t dev = (uint8_t)(EEPROM_I2C_ADDR << 1);
    if (len == 0U) {
        return 1U;
    }

    i2c_start();
    if (!i2c_write_byte((uint8_t)(dev | 0U))) {
        i2c_stop();
        return 0U;
    }
    if (!i2c_write_byte((uint8_t)(addr >> 8))) {
        i2c_stop();
        return 0U;
    }
    if (!i2c_write_byte((uint8_t)(addr & 0xFFU))) {
        i2c_stop();
        return 0U;
    }

    i2c_start();
    if (!i2c_write_byte((uint8_t)(dev | 1U))) {
        i2c_stop();
        return 0U;
    }

    for (uint16_t i = 0; i < len; i++) {
        uint8_t ack = (uint8_t)(i + 1U < len);
        buf[i] = i2c_read_byte(ack);
    }

    i2c_stop();
    return 1U;
}
