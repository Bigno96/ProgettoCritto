#include "Clmul.h"

#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <stdalign.h>
#include <stdint.h>
#include <inttypes.h>

void Array_Clmul(uint32_t n3, uint64_t Res[], uint32_t n1, uint64_t Vett1[],  uint32_t n2, uint64_t Vett2[]);

#define N 255

int main(int argc, char *argv[])
{
    uint32_t n1, n2, n3;
    n1 = (uint32_t)3;
    n2 = (uint32_t)3;
    n3 = n1 + n2;

    uint64_t Vett1[N], Vett2[N];
    uint64_t Res[N];


    Vett1[0] = (uint64_t)UINT64_MAX;
    Vett1[1] = (uint64_t)UINT64_MAX;
    Vett1[2] = (uint64_t)UINT64_MAX;


    Vett2[0] = (uint64_t)1;
    Vett2[1] = (uint64_t)UINT64_MAX;
    Vett2[2] = (uint64_t)UINT64_MAX;

    Array_Clmul(n3, Res, n1, Vett1, n2, Vett2);

    return 0;

}

void Array_Clmul(uint32_t n3, uint64_t Res[], uint32_t n1, uint64_t Vett1[], uint32_t n2, uint64_t Vett2[])
{
    __m256i ResTemp;
    __m256i Res256[(n3+3) >> 2];
    int64_t i=0, j=0;
    __m128i V1, V2;

    /*for(i = 0; i < n3; i++)
    {
        Res256[i] = _mm256_set_epi64x ((uint64_t)0, (uint64_t)0, (uint64_t)0, (uint64_t)0);
    }*/

    for(j = 0; j < (n2 >> 1); j++)          //Ciclo j n2/2 per difetto
    {
        V2 = _mm_set_epi64x (Vett2[2*j + 1] , Vett2[2*j]);

        for(i = 0; i < (n1 >> 1) ; i++)        //Ciclo i n1/2 per difetto
        {
            V1 = _mm_set_epi64x (Vett1[2*i + 1], Vett1[2*i]);
            ResTemp = clmul((__m128i)V1, (__m128i)V2);     //Clmul Temporanea
            Res256[j + i] ^= ResTemp;             //Xor con Res e Temp in Res
        }
    }

    if(n1 % 2 == 1)          //n1 dispari
    {
        V1 = _mm_set_epi64x (0, Vett1[n1 - 1]);

        for(i = 0; i < (n2 >> 1); i++)
        {
            V2 = _mm_set_epi64x (Vett2[2*i+1], Vett2[2*i]);
            ResTemp = clmul((__m128i)V1, (__m128i)V2);
            Res256[i + ((n1 + 1) >> 1) - 1] ^= ResTemp;
        }
    }


    if(n2 % 2 == 1)         //n2 dispari
    {
        V2 = _mm_set_epi64x (0, Vett2[n2-1]);

        for(i = 0; i < (n1 >> 1); i++)
        {
            V1 = _mm_set_epi64x (Vett1[2*i+1], Vett1[2*i]);
            ResTemp = clmul((__m128i)V1, (__m128i)V2);
            Res256[i + ((n2 + 1) >> 1) - 1] ^= ResTemp;
        }
    }

    if((n2 % 2 == 1) && (n1 % 2 == 1))        //n1 e n2 dispari
    {
        V1 = _mm_set_epi64x (0, Vett1[n1-1]);
        V2 = _mm_set_epi64x (0, Vett2[n2-1]);
        ResTemp = clmul((__m128i)V1, (__m128i)V2);
        Res256[((n3+3) >> 2) - 1] ^= ResTemp;
    }

    alignas (32) uint64_t v[4];

    for (j = 0; j < (n3+3) >> 2; j++)
    {
        _mm256_store_si256((__m256i*)v, Res256[j]);
        for (i = 0; i < 4; i++)
        {
            print__m256(Res256[j]);
            printf("[%ld] %016lX\n", i, v[i]);
            printf("\n\n");
        }
    }

}
