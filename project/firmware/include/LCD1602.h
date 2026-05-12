#ifndef LCD1602_H
#define LCD1602_H

/*
 * HD44780 4-bit:
 * Data: D4=PC4, D5=PC5, D6=PC6, D7=PC7
 * Control: EN=PC0, RW=PD5, RS=PD6
 */
#define LCD_DATA_PORT PORTC
#define LCD_DATA_PIN  PINC
#define LCD_DATA_DDR  DDRC

#define LCD_CTRL_PORT PORTD
#define LCD_CTRL_DDR  DDRD

#define LCD_EN PC0
#define LCD_RW PD5
#define LCD_RS PD6

#define LCD_DATA_MASK ((1 << PC4) | (1 << PC5) | (1 << PC6) | (1 << PC7))

void LCD_Init(void);
void LCD_Command(unsigned char cmnd);
void LCD_Char(unsigned char data);
void LCD_String(char *str);
void LCD_String_xy(char row, char pos, char *str);
void LCD_Clear(void);
void LCD_Gotoxy(char row, char pos);
void lcd_create_char(unsigned char address, unsigned char pattern[]);
unsigned char LCD_Read_Char(unsigned char address);

#endif
