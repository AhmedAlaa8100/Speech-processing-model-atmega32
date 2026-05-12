/* Defaults — overwritten by `training/export.py` after PTQ. */
#ifndef KWS_MODEL_SCALES_GENERATED_H
#define KWS_MODEL_SCALES_GENERATED_H

#define kws_input_scale (1.0e-2f)
#define kws_w1_scale (1.0e-3f)
#define kws_hidden_scale (1.0e-2f)

/* Integer FC1 path: hidden ~ acc * (s_x*s_w1/s_h), see export.py */
#define KWS_FC1_RESCALE_MUL (1049L)
#define KWS_FC1_RESCALE_SHR 20

#endif
