/*
 * main.c  -  Isolated-Word Speech Recognition on ATmega32A
 * ---------------------------------------------------------
 * Hardware:
 *   Microphone : MAX9814 output -> PA0 (ADC0)
 *   LCD        : HD44780 4-bit mode on PORTC
 *                EN=PC0, RW=PC1, RS=PC2, D4=PC4, D5=PC5, D6=PC6, D7=PC7
 *   Trigger    : Push-button on PB0 (active LOW, internal pull-up)
 *   F_CPU      : 11059200 Hz
 *   Sample rate: 8000 Hz via Timer1 CTC (OCR1A = 1381)
 */

#define F_CPU 11059200UL

#include <avr/interrupt.h>
#include <avr/io.h>
#include <avr/pgmspace.h>
#include <stdint.h>
#include <util/delay.h>

#include "LCD1602.h"
#include "kws_model.h"
#include "mfcc.h"
#include "sram.h"

/* ================================================================
   Configuration
   ================================================================ */
#define SAMPLE_RATE        8000U
#define OCR1A_VALUE        ((uint16_t)((F_CPU / SAMPLE_RATE) - 1U))

#define FRAME_LEN_SAMPLES  200U
#define HOP_LEN_SAMPLES    80U
#define RECORD_MS          1500U
#define RECORD_SAMPLES     ((uint16_t)(((uint32_t)SAMPLE_RATE * (uint32_t)RECORD_MS) / 1000U))
#define TOTAL_FRAMES       ((uint16_t)(((RECORD_SAMPLES - FRAME_LEN_SAMPLES) / HOP_LEN_SAMPLES) + 1U))

#define AUDIO_SRAM_BASE    0x0000U

#define ENERGY_THRESHOLD   0.002f
#define ENERGY_THRESHOLD_SUM \
    ((uint32_t)(ENERGY_THRESHOLD * (float)FRAME_LEN_SAMPLES * 512.0f * 512.0f))
#define SILENCE_FRAMES     5U

#define ADC_CHANNEL        0U

#define BTN_PIN   PINB
#define BTN_PORT  PORTB
#define BTN_DDR   DDRB
#define BTN_BIT   PB0

volatile uint8_t recording = 0;
volatile uint32_t sample_count = 0;

static int8_t input_q[KWS_INPUT_SIZE];
static int8_t hidden_q[KWS_HIDDEN_SIZE];
static int32_t logits[KWS_NUM_CLASSES];
static uint32_t window_energy[KWS_NUM_FRAMES];
static uint8_t window_voiced[KWS_NUM_FRAMES];

/* Class index order must match training export: up, down, left, right, yes, no, on, off */
static const char label_0[] PROGMEM = "up";
static const char label_1[] PROGMEM = "down";
static const char label_2[] PROGMEM = "left";
static const char label_3[] PROGMEM = "right";
static const char label_4[] PROGMEM = "yes";
static const char label_5[] PROGMEM = "no";
static const char label_6[] PROGMEM = "on";
static const char label_7[] PROGMEM = "off";

static const char *const kws_labels[KWS_NUM_CLASSES] PROGMEM = {
    label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7,
};

void adc_init(void)
{
    DDRA &= (uint8_t) ~(1U << ADC_CHANNEL);
    PORTA &= (uint8_t) ~(1U << ADC_CHANNEL);
    ADMUX = (uint8_t)((1U << REFS0) | ADC_CHANNEL);
    ADCSRA = (uint8_t)((1U << ADEN) | (1U << ADPS2) | (1U << ADPS1));
}

uint16_t adc_read(void)
{
    ADCSRA |= (uint8_t)(1U << ADSC);
    while (ADCSRA & (uint8_t)(1U << ADSC)) {
    }
    return ADC;
}

void timer1_init(void)
{
    TCCR1B = (uint8_t)((1U << WGM12) | (1U << CS10));
    OCR1A = OCR1A_VALUE;
    TIMSK &= (uint8_t) ~(1U << OCIE1A);
}

ISR(TIMER1_COMPA_vect)
{
    if (!recording) {
        return;
    }

    uint16_t raw = adc_read();
    int16_t s = (int16_t)raw - 512;

    if (sample_count < RECORD_SAMPLES) {
        sram_write_u16(s);
        sample_count++;
    }

    if (sample_count >= RECORD_SAMPLES) {
        recording = 0;
        sram_end_write();
        TIMSK &= (uint8_t) ~(1U << OCIE1A);
    }
}

static int8_t clamp_int8(int32_t x)
{
    if (x > 127) {
        return 127;
    }
    if (x < -128) {
        return -128;
    }
    return (int8_t)x;
}

static uint16_t find_best_window_sram(uint16_t base_addr, uint8_t *out_voiced)
{
    if (TOTAL_FRAMES < KWS_NUM_FRAMES) {
        if (out_voiced) {
            *out_voiced = 0U;
        }
        return 0U;
    }

    uint64_t win_sum = 0;
    uint16_t win_voiced = 0;

    for (uint8_t f = 0; f < KWS_NUM_FRAMES; f++) {
        uint16_t addr = (uint16_t)(base_addr + (uint32_t)((uint16_t)f * HOP_LEN_SAMPLES) * 2U);
        uint32_t sum = 0U;

        sram_begin_read(addr);
        for (uint16_t i = 0; i < FRAME_LEN_SAMPLES; i++) {
            int16_t ss = sram_read_u16();
            sum += (uint32_t)((int32_t)ss * ss);
        }
        sram_end_read();

        window_energy[f] = sum;
        window_voiced[f] = (sum > ENERGY_THRESHOLD_SUM) ? 1U : 0U;
        win_sum += sum;
        win_voiced += window_voiced[f];
    }

    uint16_t best_start = 0;
    uint64_t best_sum = win_sum;
    uint16_t best_voiced = win_voiced;

    for (uint16_t f = KWS_NUM_FRAMES; f < TOTAL_FRAMES; f++) {
        uint8_t idx = (uint8_t)(f % KWS_NUM_FRAMES);

        win_sum -= window_energy[idx];
        win_voiced -= window_voiced[idx];

        uint16_t start = (uint16_t)((uint32_t)f * HOP_LEN_SAMPLES);
        uint16_t addr = (uint16_t)(base_addr + (uint32_t)start * 2U);
        uint32_t sum = 0U;

        sram_begin_read(addr);
        for (uint16_t i = 0; i < FRAME_LEN_SAMPLES; i++) {
            int16_t ss = sram_read_u16();
            sum += (uint32_t)((int32_t)ss * ss);
        }
        sram_end_read();

        window_energy[idx] = sum;
        window_voiced[idx] = (sum > ENERGY_THRESHOLD_SUM) ? 1U : 0U;
        win_sum += sum;
        win_voiced += window_voiced[idx];

        uint16_t start_frame = (uint16_t)(f - KWS_NUM_FRAMES + 1U);
        if (win_sum > best_sum) {
            best_sum = win_sum;
            best_start = start_frame;
            best_voiced = win_voiced;
        }
    }

    if (out_voiced) {
        *out_voiced = (uint8_t)((best_voiced > 255U) ? 255U : best_voiced);
    }
    return best_start;
}

static void kws_run_inference(const int8_t *in_q, int32_t out_logits[KWS_NUM_CLASSES])
{
    for (uint8_t i = 0; i < KWS_HIDDEN_SIZE; i++) {
        int32_t acc = (int32_t)pgm_read_dword(&kws_fc1_b[i]);
        for (uint16_t k = 0; k < KWS_INPUT_SIZE; k++) {
            int8_t w = (int8_t)pgm_read_byte(&kws_fc1_w[i][k]);
            acc += (int32_t)w * (int32_t)in_q[k];
        }
        int64_t t = (int64_t)acc * (int64_t)KWS_FC1_RESCALE_MUL;
        int32_t acc_scaled = (int32_t)(t >> KWS_FC1_RESCALE_SHR);
        if (acc_scaled < 0) {
            acc_scaled = 0;
        }
        hidden_q[i] = clamp_int8(acc_scaled);
    }

    for (uint8_t c = 0; c < KWS_NUM_CLASSES; c++) {
        int32_t acc = (int32_t)pgm_read_dword(&kws_fc2_b[c]);
        for (uint8_t i = 0; i < KWS_HIDDEN_SIZE; i++) {
            int8_t w = (int8_t)pgm_read_byte(&kws_fc2_w[c][i]);
            acc += (int32_t)w * (int32_t)hidden_q[i];
        }
        out_logits[c] = acc;
    }
}

static uint8_t kws_argmax(const int32_t in_logits[KWS_NUM_CLASSES])
{
    uint8_t best = 0;
    int32_t best_val = in_logits[0];
    for (uint8_t i = 1; i < KWS_NUM_CLASSES; i++) {
        if (in_logits[i] > best_val) {
            best_val = in_logits[i];
            best = i;
        }
    }
    return best;
}

static uint8_t kws_confidence_pct(const int32_t in_logits[KWS_NUM_CLASSES], uint8_t best)
{
    if (in_logits[best] <= 0) {
        return 0U;
    }

    int32_t sum = 0;
    for (uint8_t i = 0; i < KWS_NUM_CLASSES; i++) {
        if (in_logits[i] > 0) {
            sum += in_logits[i];
        }
    }

    if (sum <= 0) {
        return 0U;
    }
    return (uint8_t)(((int32_t)in_logits[best] * 100 + (sum / 2)) / sum);
}

static void lcd_print_u8_dec(uint8_t v)
{
    char buf[4];
    uint8_t idx = 0;
    if (v >= 100U) {
        buf[idx++] = (char)('0' + (v / 100U));
        v = (uint8_t)(v % 100U);
        buf[idx++] = (char)('0' + (v / 10U));
        buf[idx++] = (char)('0' + (v % 10U));
    } else if (v >= 10U) {
        buf[idx++] = (char)('0' + (v / 10U));
        buf[idx++] = (char)('0' + (v % 10U));
    } else {
        buf[idx++] = (char)('0' + v);
    }
    buf[idx] = '\0';
    LCD_String(buf);
}

static void lcd_print_u32_dec(uint32_t v)
{
    char buf[11];
    uint8_t idx = 0;

    if (v == 0U) {
        LCD_String("0");
        return;
    }

    while (v > 0U && idx < (uint8_t)(sizeof(buf) - 1U)) {
        buf[idx++] = (char)('0' + (v % 10U));
        v /= 10U;
    }

    while (idx > 0U) {
        LCD_Char((unsigned char)buf[--idx]);
    }
}

static uint32_t compute_max_energy(uint16_t base_addr, uint16_t start_frame)
{
    uint32_t max_sum = 0U;
    for (uint8_t f = 0; f < KWS_NUM_FRAMES; f++) {
        uint16_t start = (uint16_t)(start_frame + f);
        uint16_t addr = (uint16_t)(base_addr + (uint32_t)start * HOP_LEN_SAMPLES * 2U);
        uint32_t sum = 0U;

        sram_begin_read(addr);
        for (uint16_t i = 0; i < FRAME_LEN_SAMPLES; i++) {
            int16_t ss = sram_read_u16();
            sum += (uint32_t)((int32_t)ss * ss);
        }
        sram_end_read();

        if (sum > max_sum) {
            max_sum = sum;
        }
    }
    return max_sum;
}

void lcd_show_idle(void)
{
    LCD_Clear();
    LCD_String_xy(0, 0, "Speech Recog v1 ");
    LCD_String_xy(1, 0, "Press BTN->Speak");
}

void lcd_show_recording(void)
{
    LCD_Clear();
    LCD_String_xy(0, 0, "* RECORDING...  ");
    LCD_String_xy(1, 0, "Please speak now");
}

void lcd_show_processing(void)
{
    LCD_Clear();
    LCD_String_xy(0, 0, "  Processing... ");
    LCD_String_xy(1, 0, "                ");
}

void lcd_show_result(uint8_t word_idx, uint8_t score_pct)
{
    char word_buf[8];
    const char *ptr = (const char *)pgm_read_word(&kws_labels[word_idx]);
    uint8_t i = 0;
    char c;
    while ((c = pgm_read_byte(ptr++)) && i < 7) {
        word_buf[i++] = c;
    }
    word_buf[i] = '\0';

    LCD_Clear();
    LCD_String_xy(0, 0, "Word: ");
    LCD_String(word_buf);

    LCD_String_xy(1, 0, "Conf: ");
    lcd_print_u8_dec(score_pct);
    LCD_String("%   ");
}

void lcd_show_silence(void)
{
    LCD_Clear();
    LCD_String_xy(0, 0, "No speech found!");
    LCD_String_xy(1, 0, "Try again...    ");
}

void lcd_show_mfcc_fail(void)
{
    LCD_Clear();
    LCD_String_xy(0, 0, "MFCC error!     ");
    LCD_String_xy(1, 0, "Check microphone");
}

void start_recording(void)
{
    cli();
    sample_count = 0;
    TCNT1 = 0;
    TIFR |= (uint8_t)(1U << OCF1A);
    sram_begin_write(AUDIO_SRAM_BASE);
    recording = 1;
    TIMSK |= (uint8_t)(1U << OCIE1A);
    sei();
}

void wait_recording_done(void)
{
    uint16_t guard = (uint16_t)((RECORD_MS / 10U) + 50U);
    while (recording && guard--) {
        _delay_ms(10);
    }

    if (recording) {
        cli();
        recording = 0;
        sram_end_write();
        TIMSK &= (uint8_t) ~(1U << OCIE1A);
        sei();
    }
}

void hw_init(void)
{
    BTN_DDR &= (uint8_t) ~(1U << BTN_BIT);
    BTN_PORT |= (uint8_t)(1U << BTN_BIT);

    adc_init();
    sram_init();
    LCD_Init();
    timer1_init();
    sei();
}

int main(void)
{
    hw_init();
    lcd_show_idle();

    for (;;) {
        if (!(BTN_PIN & (1U << BTN_BIT))) {
            _delay_ms(30);
            if (BTN_PIN & (1U << BTN_BIT)) {
                continue;
            }

            lcd_show_recording();

            start_recording();
            wait_recording_done();

            lcd_show_processing();

            uint8_t voiced_frames = 0;
            uint16_t best_frame = find_best_window_sram(AUDIO_SRAM_BASE, &voiced_frames);
            uint16_t start_sample = (uint16_t)((uint32_t)best_frame * HOP_LEN_SAMPLES);
            uint32_t max_energy = compute_max_energy(AUDIO_SRAM_BASE, best_frame);

            LCD_Clear();
            LCD_String_xy(0, 0, "V:");
            lcd_print_u8_dec(voiced_frames);
            LCD_String(" E:");
            lcd_print_u32_dec(max_energy >> 10);
            LCD_String_xy(1, 0, "T:");
            lcd_print_u32_dec((uint32_t)(ENERGY_THRESHOLD_SUM >> 10));
            _delay_ms(1200);

            if (voiced_frames < SILENCE_FRAMES) {
                lcd_show_silence();
                _delay_ms(2000);
            } else if (!mfcc_compute_sram_to_int8(
                           (uint16_t)(AUDIO_SRAM_BASE + (uint32_t)start_sample * 2U),
                           (uint16_t)(RECORD_SAMPLES - start_sample),
                           input_q)) {
                lcd_show_mfcc_fail();
                _delay_ms(2000);
            } else {
                kws_run_inference(input_q, logits);
                uint8_t best = kws_argmax(logits);
                uint8_t score = kws_confidence_pct(logits, best);
                lcd_show_result(best, score);
                _delay_ms(3000);
            }

            lcd_show_idle();

            while (!(BTN_PIN & (1U << BTN_BIT))) {
                _delay_ms(10);
            }
            _delay_ms(50);
        }

        _delay_ms(10);
    }

    return 0;
}
