#ifndef F_CPU
#define F_CPU 11059200UL
#endif

#include <avr/io.h>
#include <stdint.h>
#include <util/delay.h>

#include "LCD1602.h"

void LCD_Command(unsigned char cmnd)
{
    LCD_CTRL_PORT &= (uint8_t) ~(1 << LCD_RW);
    LCD_CTRL_PORT &= (uint8_t) ~(1 << LCD_RS);
    LCD_DATA_PORT = (LCD_DATA_PORT & (uint8_t)~LCD_DATA_MASK) | (cmnd & 0xF0U);
    LCD_DATA_PORT |= (uint8_t)(1 << LCD_EN);
    _delay_us(1);
    LCD_DATA_PORT &= (uint8_t) ~(1 << LCD_EN);

    _delay_us(200);

    LCD_DATA_PORT = (LCD_DATA_PORT & (uint8_t)~LCD_DATA_MASK) | (unsigned char)(cmnd << 4);
    LCD_DATA_PORT |= (uint8_t)(1 << LCD_EN);
    _delay_us(1);
    LCD_DATA_PORT &= (uint8_t) ~(1 << LCD_EN);
    _delay_ms(2);
}

void LCD_Char(unsigned char data)
{
    LCD_CTRL_PORT &= (uint8_t) ~(1 << LCD_RW);
    LCD_CTRL_PORT |= (uint8_t)(1 << LCD_RS);
    LCD_DATA_PORT = (LCD_DATA_PORT & (uint8_t)~LCD_DATA_MASK) | (data & 0xF0U);

    LCD_DATA_PORT |= (uint8_t)(1 << LCD_EN);
    _delay_us(1);
    LCD_DATA_PORT &= (uint8_t) ~(1 << LCD_EN);

    _delay_us(200);

    LCD_DATA_PORT = (LCD_DATA_PORT & (uint8_t)~LCD_DATA_MASK) | (unsigned char)(data << 4);
    LCD_DATA_PORT |= (uint8_t)(1 << LCD_EN);
    _delay_us(1);
    LCD_DATA_PORT &= (uint8_t) ~(1 << LCD_EN);
    _delay_ms(2);
}

void LCD_Init(void)
{
    LCD_DATA_DDR |= (uint8_t)((1 << LCD_EN) | LCD_DATA_MASK);
    LCD_CTRL_DDR |= (uint8_t)((1 << LCD_RW) | (1 << LCD_RS));
    LCD_CTRL_PORT &= (uint8_t)~((1 << LCD_RW) | (1 << LCD_RS));
    LCD_DATA_PORT &= (uint8_t) ~(1 << LCD_EN);
    _delay_ms(20);

    LCD_Command(0x02U);
    LCD_Command(0x28U);
    LCD_Command(0x0cU);
    LCD_Command(0x06U);
    LCD_Command(0x01U);
    _delay_ms(2);
}

void LCD_String(char *str)
{
    for (int i = 0; str[i] != 0; i++) {
        LCD_Char((unsigned char)str[i]);
    }
}

void LCD_String_xy(char row, char pos, char *str)
{
    if (row == 0 && pos < 16) {
        LCD_Command((unsigned char)((pos & 0x0FU) | 0x80U));
    } else if (row == 1 && pos < 16) {
        LCD_Command((unsigned char)((pos & 0x0FU) | 0xC0U));
    }
    LCD_String(str);
}

void LCD_Clear(void)
{
    LCD_Command(0x01U);
    _delay_ms(2);
    LCD_Command(0x80U);
}

void lcd_create_char(unsigned char address, unsigned char pattern[])
{
    if (address < 0x40U || address > 0x78U) {
        return;
    }
    LCD_Command(address);
    for (uint8_t i = 0; i < 8U; i++) {
        LCD_Char(pattern[i]);
    }
}

void LCD_Gotoxy(char row, char pos)
{
    if (row == 0 && pos < 16) {
        LCD_Command((unsigned char)((pos & 0x0FU) | 0x80U));
    } else if (row == 1 && pos < 16) {
        LCD_Command((unsigned char)((pos & 0x0FU) | 0xC0U));
    }
}

unsigned char LCD_Read_Char(unsigned char address)
{
    unsigned char value = 0;
    LCD_Command(address);

    LCD_DATA_DDR &= (uint8_t)~LCD_DATA_MASK;
    LCD_DATA_DDR |= (uint8_t)(1 << LCD_EN);

    LCD_CTRL_PORT |= (uint8_t)(1 << LCD_RS);
    LCD_CTRL_PORT |= (uint8_t)(1 << LCD_RW);
    LCD_DATA_PORT |= (uint8_t)(1 << LCD_EN);

    _delay_us(20);
    value = (unsigned char)(LCD_DATA_PIN & 0xF0U);
    LCD_DATA_PORT &= (uint8_t) ~(1 << LCD_EN);
    _delay_us(200);

    LCD_DATA_PORT |= (uint8_t)(1 << LCD_EN);
    _delay_us(20);
    value = (unsigned char)(value | ((LCD_DATA_PIN & 0xF0U) >> 4));
    LCD_DATA_PORT &= (uint8_t) ~(1 << LCD_EN);

    LCD_DATA_DDR |= LCD_DATA_MASK;
    LCD_CTRL_PORT &= (uint8_t) ~(1 << LCD_RW);
    return value;
}
