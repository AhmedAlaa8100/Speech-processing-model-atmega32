#ifndef CONFIG_H
#define CONFIG_H

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

#endif
