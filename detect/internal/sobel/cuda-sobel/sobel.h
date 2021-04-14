#ifndef SOBEL_H
#define SOBEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int ApplySobel(uint32_t *data, int w, int h);

#ifdef __cplusplus
}
#endif

#endif
