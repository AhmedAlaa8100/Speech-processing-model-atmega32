#ifndef LCD1602_H
#define LCD1602_H

/*
 * HD44780 4-bit on PORTC:
 * EN=PC0, RW=PC1, RS=PC2, D4=PC4, D5=PC5, D6=PC6, D7=PC7
 */
#define LCD_Port PORTC
#define LCD_Pin  PINC
#define LCD_Dir  DDRC

#define EN PC0
#define RW PC1
#define RS PC2

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
