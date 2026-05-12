#ifndef MFCC_H
#define MFCC_H

#include <stdint.h>

#include "kws_model.h"

#define MFCC_N_FFT       256U
#define MFCC_WIN_LENGTH  200U
#define MFCC_HOP_LENGTH   80U
#define MFCC_N_MELS       26U
#define MFCC_LOG2_Q       12U
#define MFCC_POWER_SHIFT   8U
#define MFCC_LN2_Q      2838U /* round(ln(2) * 4096) */

/* Fixed-point MFCC -> int8 (must match `export.py` / kws_mfcc_i8_mul tables) */
#define MFCC_NORM_SHR   20

uint8_t mfcc_compute_sram_to_int8(uint16_t base_addr, uint16_t num_samples, int8_t *out_q);

#endif
