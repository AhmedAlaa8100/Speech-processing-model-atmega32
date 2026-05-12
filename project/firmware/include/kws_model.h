#ifndef KWS_MODEL_H
#define KWS_MODEL_H

#include <avr/pgmspace.h>
#include <stdint.h>

#include "kws_model_scales.generated.h"

#define KWS_NUM_CLASSES  8U
#define KWS_NUM_FRAMES   16U
#define KWS_N_MFCC       13U
#define KWS_INPUT_SIZE   ((uint16_t)(KWS_N_MFCC * KWS_NUM_FRAMES))
#define KWS_HIDDEN_SIZE  96U

#ifndef KWS_USE_EEPROM_WEIGHTS
extern const int8_t kws_fc1_w[KWS_HIDDEN_SIZE][KWS_INPUT_SIZE] PROGMEM;
extern const int32_t kws_fc1_b[KWS_HIDDEN_SIZE] PROGMEM;
extern const int8_t kws_fc2_w[KWS_NUM_CLASSES][KWS_HIDDEN_SIZE] PROGMEM;
extern const int32_t kws_fc2_b[KWS_NUM_CLASSES] PROGMEM;
#endif

extern const float kws_input_mean[KWS_N_MFCC] PROGMEM;
extern const float kws_input_std[KWS_N_MFCC] PROGMEM;

/* Integer MFCC path (see export.py) — avoids soft-float in mfcc.c */
extern const int32_t kws_mfcc_mean_q12[KWS_N_MFCC] PROGMEM;
extern const int32_t kws_mfcc_i8_mul[KWS_N_MFCC] PROGMEM;

#endif
