#ifndef ACLMULK_H
#define ACLMULK_H

#include <inttypes.h>
#define DIGIT uint64_t

/*
 * Carry less multiplication between two arrays, reduction time with Karatsuba's algorithm
 * Result sets in Res
 */
void ACK(const uint32_t nRes, DIGIT Res[], const uint32_t n1, const DIGIT Vect1[], const uint32_t n2, const DIGIT Vect2[]);

#endif 

