#include <avr/pgmspace.h>
#include <stdint.h>

#include "mfcc.h"
#include "mfcc_tables.h"
#include "sram.h"

#define FFT_LEN           MFCC_N_FFT
#define FRAME_LEN_SAMPLES MFCC_WIN_LENGTH
#define HOP_LEN_SAMPLES   MFCC_HOP_LENGTH

#define FFT_STAGE_SHIFT 1
#define FFT_STAGES      8
#define FFT_SCALE_SHIFT (FFT_STAGE_SHIFT * FFT_STAGES)

#define INPUT_SHIFT           6
#define INPUT_LOG2_CORRECTION (2U * INPUT_SHIFT)

#define DC_FILTER_SHIFT 5

static int16_t fft_real[FFT_LEN];
static uint32_t mel_energy[MFCC_N_MELS];

static int16_t clamp_int16(int32_t v)
{
    if (v > 32767) {
        return 32767;
    }
    if (v < -32768) {
        return -32768;
    }
    return (int16_t)v;
}

static int8_t clamp_int8(int32_t v)
{
    if (v > 127) {
        return 127;
    }
    if (v < -128) {
        return -128;
    }
    return (int8_t)v;
}

static uint32_t log2_approx_q12(uint32_t x)
{
    if (x == 0U) {
        return 0U;
    }

    uint8_t msb = 0;
    uint32_t tmp = x;
    while (tmp >>= 1U) {
        msb++;
    }

    uint32_t mant = x - (1UL << msb);
    uint8_t idx;
    if (msb >= 8U) {
        idx = (uint8_t)(mant >> (msb - 8U));
    } else {
        idx = (uint8_t)(mant << (8U - msb));
    }

    uint32_t frac = (uint32_t)pgm_read_word(&kws_log2_lut_q12[idx]);
    return ((uint32_t)msb << MFCC_LOG2_Q) + frac;
}

static void fft_radix2_int16(int16_t *real, int16_t *imag)
{
    uint16_t j = 0;
    for (uint16_t i = 1; i < FFT_LEN - 1U; i++) {
        uint16_t bit = FFT_LEN >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1U;
        }
        j ^= bit;
        if (i < j) {
            int16_t tr = real[i];
            real[i] = real[j];
            real[j] = tr;
            int16_t ti = imag[i];
            imag[i] = imag[j];
            imag[j] = ti;
        }
    }

    for (uint16_t len = 2; len <= FFT_LEN; len <<= 1U) {
        uint16_t half = len >> 1U;
        uint16_t step = FFT_LEN / len;
        for (uint16_t i = 0; i < FFT_LEN; i += len) {
            for (uint16_t j2 = 0; j2 < half; j2++) {
                uint16_t k = j2 * step;
                int16_t wr = (int16_t)pgm_read_word(&kws_twiddle_cos_q15[k]);
                int16_t wi = (int16_t)pgm_read_word(&kws_twiddle_sin_q15[k]);

                uint16_t idx = i + j2;
                uint16_t idx2 = idx + half;
                int32_t r2 = real[idx2];
                int32_t i2 = imag[idx2];

                int32_t tr = (int32_t)(((int64_t)wr * r2 - (int64_t)wi * i2) >> 15);
                int32_t ti = (int32_t)(((int64_t)wr * i2 + (int64_t)wi * r2) >> 15);

                int32_t r_sum = ((int32_t)real[idx] + tr) >> FFT_STAGE_SHIFT;
                int32_t i_sum = ((int32_t)imag[idx] + ti) >> FFT_STAGE_SHIFT;
                int32_t r_dif = ((int32_t)real[idx] - tr) >> FFT_STAGE_SHIFT;
                int32_t i_dif = ((int32_t)imag[idx] - ti) >> FFT_STAGE_SHIFT;

                real[idx] = clamp_int16(r_sum);
                imag[idx] = clamp_int16(i_sum);
                real[idx2] = clamp_int16(r_dif);
                imag[idx2] = clamp_int16(i_dif);
            }
        }
    }
}

uint8_t mfcc_compute_sram_to_int8(uint16_t base_addr, uint16_t num_samples, int8_t *out_q)
{
    int16_t fft_imag[FFT_LEN];

    if (out_q == 0) {
        return 0U;
    }

    uint16_t needed = FRAME_LEN_SAMPLES + (uint16_t)((KWS_NUM_FRAMES - 1U) * HOP_LEN_SAMPLES);
    if (num_samples < needed) {
        return 0U;
    }

    for (uint8_t frame = 0; frame < KWS_NUM_FRAMES; frame++) {
        uint16_t start = (uint16_t)frame * HOP_LEN_SAMPLES;
        uint16_t addr = (uint16_t)(base_addr + (uint32_t)start * 2U);
        int32_t dc = 0;

        sram_begin_read(addr);
        for (uint16_t i = 0; i < FRAME_LEN_SAMPLES; i++) {
            int16_t sample = sram_read_u16();

            dc += ((int32_t)sample - dc) >> DC_FILTER_SHIFT;
            int32_t hp = (int32_t)sample - dc;

            int32_t s = hp << INPUT_SHIFT;
            if (s > 32767) {
                s = 32767;
            }
            if (s < -32768) {
                s = -32768;
            }

            int16_t window = (int16_t)pgm_read_word(&kws_window_q15[i]);
            int32_t win = (int32_t)(((int64_t)s * window) >> 15);
            fft_real[i] = clamp_int16(win);
            fft_imag[i] = 0;
        }
        sram_end_read();

        for (uint16_t i = FRAME_LEN_SAMPLES; i < FFT_LEN; i++) {
            fft_real[i] = 0;
            fft_imag[i] = 0;
        }

        fft_radix2_int16(fft_real, fft_imag);

        for (uint8_t m = 0; m < MFCC_N_MELS; m++) {
            mel_energy[m] = 0U;
        }

        for (uint16_t k = 0; k < FFT_BINS; k++) {
            int32_t re = fft_real[k];
            int32_t im = fft_imag[k];
            uint64_t p = (uint64_t)((int64_t)re * re + (int64_t)im * im);

            uint32_t p_scaled = (uint32_t)(p >> MFCC_POWER_SHIFT);
            if (p_scaled == 0U) {
                p_scaled = 1U;
            }

            for (uint8_t m = 0; m < MFCC_N_MELS; m++) {
                uint16_t w = (uint16_t)pgm_read_word(&kws_mel_filters_q15[m][k]);
                mel_energy[m] += (uint32_t)(((uint64_t)p_scaled * w) >> 15);
            }
        }

        int32_t acc[KWS_N_MFCC];
        for (uint8_t c = 0; c < KWS_N_MFCC; c++) {
            acc[c] = 0;
        }

        for (uint8_t m = 0; m < MFCC_N_MELS; m++) {
            uint32_t val = mel_energy[m];
            if (val == 0U) {
                val = 1U;
            }

            uint32_t log2_q12 = log2_approx_q12(val);
            log2_q12 += (uint32_t)(MFCC_POWER_SHIFT + (2U * FFT_SCALE_SHIFT) + INPUT_LOG2_CORRECTION) << MFCC_LOG2_Q;

            uint32_t ln_q12 = (log2_q12 * (uint32_t)MFCC_LN2_Q) >> MFCC_LOG2_Q;

            for (uint8_t c = 0; c < KWS_N_MFCC; c++) {
                int16_t dct = (int16_t)pgm_read_word(&kws_dct_q15[c][m]);
                acc[c] += (int32_t)dct * (int32_t)ln_q12;
            }
        }

        for (uint8_t c = 0; c < KWS_N_MFCC; c++) {
            int32_t mfcc_q12 = acc[c] >> 15;
            int32_t mean_q = (int32_t)pgm_read_dword(&kws_mfcc_mean_q12[c]);
            int32_t mul = (int32_t)pgm_read_dword(&kws_mfcc_i8_mul[c]);
            int32_t diff = mfcc_q12 - mean_q;
            int64_t pr = (int64_t)diff * (int64_t)mul;
            int32_t sf = (int32_t)((pr + ((int64_t)1 << (MFCC_NORM_SHR - 1))) >> MFCC_NORM_SHR);
            out_q[(uint16_t)c * KWS_NUM_FRAMES + frame] = clamp_int8(sf);
        }
    }

    return 1U;
}
