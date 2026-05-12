#ifndef MFCC_TABLES_H
#define MFCC_TABLES_H

#include <stdint.h>
#include <avr/pgmspace.h>

#include "mfcc.h"

#define FFT_BINS ((MFCC_N_FFT) / 2U + 1U)

extern const int16_t kws_twiddle_cos_q15[128] PROGMEM;
extern const int16_t kws_twiddle_sin_q15[128] PROGMEM;
extern const int16_t kws_window_q15[MFCC_WIN_LENGTH] PROGMEM;
extern const uint16_t kws_log2_lut_q12[256] PROGMEM;
extern const uint16_t kws_mel_filters_q15[MFCC_N_MELS][FFT_BINS] PROGMEM;
extern const int16_t kws_dct_q15[KWS_N_MFCC][MFCC_N_MELS] PROGMEM;

#endif
