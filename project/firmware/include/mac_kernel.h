#ifndef MAC_KERNEL_H
#define MAC_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Signed 8-bit dot product, int32 accumulator (preferred). */
int32_t mac_kernel_dot_i32(const int8_t *a, const int8_t *b, uint8_t n);

/* int16 accumulator — may overflow for large n. */
int16_t mac_kernel_dot_i16(const int8_t *a, const int8_t *b, uint8_t n);

#ifdef __cplusplus
}
#endif

#endif
