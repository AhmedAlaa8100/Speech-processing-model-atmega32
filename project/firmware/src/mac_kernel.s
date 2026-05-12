/*
 * mac_kernel.s — ATmega32A AVR-GCC assembly
 *
 * Signed 8-bit dot product inner loop using MULS (2 cycles on classic AVR).
 * int32_t mac_kernel_dot_i32(const int8_t *a, const int8_t *b, uint8_t n);
 *
 * ABI (avr-gcc): a in r25:r24, b in r23:r22, n in r20
 * Return int32_t in r25 r24 r23 r22 (MSB .. LSB)
 *
 * Clobbers: r0 r1 r18-r21 r26-r27 r30 r31 (call-used / scratch)
 * Preserves: r2-r17, r28-r29 (Y) per typical noninterrupt leaf if we avoid Y — we use only Z.
 */

        .file   "mac_kernel.s"
        .text

        .global mac_kernel_dot_i32
        .type   mac_kernel_dot_i32, @function

mac_kernel_dot_i32:
        movw    r30, r24        /* Z = a */
        movw    r26, r22        /* X = b (use X; Y unused) */

        clr     r22             /* sum 32-bit, little-endian return layout */
        clr     r23             /* r22 = LSB ... r25 = MSB */
        clr     r24
        clr     r25

        tst     r20
        breq    2f

1:      ld      r18, Z+
        ld      r19, X+
        muls    r18, r19        /* signed 8x8 -> r1:r0 */
        add     r22, r0
        adc     r23, r1
        /* sign-extend r1 into high 16 bits of 32-bit partial */
        clr     r0
        sbrc    r1, 7
        com     r0              /* r0 = 0x00 or 0xFF */
        add     r24, r0
        adc     r25, r0

        dec     r20
        brne    1b

2:      ret

        .size   mac_kernel_dot_i32, .-mac_kernel_dot_i32

/*
 * Optional: int16_t mac_kernel_dot_i16(const int8_t *a, const int8_t *b, uint8_t n);
 * Return in r25:r24. Risk of overflow for large n — prefer i32 variant above.
 */
        .global mac_kernel_dot_i16
        .type   mac_kernel_dot_i16, @function

mac_kernel_dot_i16:
        movw    r30, r24
        movw    r26, r22

        clr     r24
        clr     r25

        tst     r20
        breq    4f

3:      ld      r18, Z+
        ld      r19, X+
        muls    r18, r19
        add     r24, r0
        adc     r25, r1

        dec     r20
        brne    3b

4:      ret

        .size   mac_kernel_dot_i16, .-mac_kernel_dot_i16
