#ifndef ACK_H
#define ACK_H

#include <inttypes.h>
#include <emmintrin.h>
#include <immintrin.h>
#define DIGIT uint64_t

/*
 * Carry less multiplication between two arrays, reduction time with Karatsuba's algorithm
 * Result sets in Res
 */
void ack(const uint32_t nRes, DIGIT Res[], const uint32_t n1, const DIGIT Vect1[], const uint32_t n2, const DIGIT Vect2[]);

/* 
 * Prints m256i with >=c99 standard
 */
void print_m256 (const __m256i num);

/* 
 * Prints m128i with >=c99 standard
 */
void print_m128 (const __m128i num);

#endif 

