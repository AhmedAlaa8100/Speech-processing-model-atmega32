#include <avr/pgmspace.h>
#include <stdint.h>

#include "kws_model.h"

/* Placeholder weights — replace with `training/export.py` output. */
const int8_t kws_fc1_w[KWS_HIDDEN_SIZE][KWS_INPUT_SIZE] PROGMEM = {0};
const int32_t kws_fc1_b[KWS_HIDDEN_SIZE] PROGMEM = {0};
const int8_t kws_fc2_w[KWS_NUM_CLASSES][KWS_HIDDEN_SIZE] PROGMEM = {0};
const int32_t kws_fc2_b[KWS_NUM_CLASSES] PROGMEM = {0};

const float kws_input_mean[KWS_N_MFCC] PROGMEM __attribute__((aligned(4))) = {0};
const float kws_input_std[KWS_N_MFCC] PROGMEM __attribute__((aligned(4))) = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
};

/* Defaults: mean 0, std 1, kws_input_scale 1e-2 — matches export.py placeholder math */
const int32_t kws_mfcc_mean_q12[KWS_N_MFCC] PROGMEM = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
const int32_t kws_mfcc_i8_mul[KWS_N_MFCC] PROGMEM = {
    25601, 25601, 25601, 25601, 25601, 25601, 25601, 25601, 25601, 25601, 25601, 25601, 25601,
};
