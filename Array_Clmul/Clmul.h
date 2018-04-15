#ifndef _CLMUL_H_
#define _CLMUL_H_

#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>

// clmul fra num1 e num2, salvata in ris
__m256i clmul (__m128i val1, __m128i val2);
// stampa m256i
void print__m256 (__m256i num);
// stampa m128i
void print__m128 (__m128i num);

#endif
