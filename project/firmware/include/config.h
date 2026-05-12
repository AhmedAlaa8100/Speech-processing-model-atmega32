#ifndef CONFIG_H
#define CONFIG_H

#include <avr/io.h>

/* Audio / MFCC */
#define SAMPLE_RATE_HZ     16000u
#define FFT_N              256u
#define MFCC_NUM_COEFF     13u
#define MFCC_TIME_FRAMES   16u

/* VAD (RMS energy gate) — tune on device */
#define VAD_RMS_Q15_THRESH 500

/* Sliding-window voting (2/3 agree, confidence > 90% in Q0.7 or float export) */
#define KWS_WINDOW_LEN        3u
#define KWS_VOTE_MIN_AGREE    2u
#define KWS_CONF_THRESH_Q7    115 /* ~0.9 * 128 */

/* External EEPROM (24Cxx) for model weights */
#define KWS_USE_EEPROM_WEIGHTS 1
#define EEPROM_I2C_ADDR        0x50u /* 7-bit base address (A2..A0 straps) */
#define EEPROM_I2C_SDA_BIT     PC1
#define EEPROM_I2C_SCL_BIT     PC2
#define EEPROM_I2C_DELAY_US    4u

#define KWS_EEPROM_BASE        0x0000u

#endif
