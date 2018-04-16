#ifndef _NMUL_H_
#define _NMUL_H_

#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>

// restituisce int a 64bit a partire da 64 bit di un vettore, DISCENDENTE
uint64_t parse_num_64 (uint64_t vet[], uint32_t pos);
// riempie 64 bit di un vettore con un int a 64bit, dalla pos specificata, DISCENDENTE
void parse_vet_64 (uint64_t num, uint64_t vet[], uint32_t pos);
// clmul fra num1 e num2, salvata in ris
void normal_mul (__m128i *num1, __m128i *num2, __m256i *ris);

#endif
