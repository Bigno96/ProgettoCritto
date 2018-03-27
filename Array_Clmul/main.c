#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <stdalign.h>
#include <stdint.h>
#include <inttypes.h>
#include "Clmul.h"

void Array_Clmul(uint64_t Vett1[], uint64_t Vett2[], uint32_t n1, uint32_t n2, __m256i Res[], uint32_t n3);

#define N 255
int main(int argc, char *argv[]) {

    uint32_t n1, n2, n3;
    n1 = 4;
    n2 = 3;
    n3 = (n1 + 1 >> 1) + (n2 + 1 >> 1) - 1;

    uint64_t Vett1[N], Vett2[N];
    __m256i Res[N];


    Vett1[0] = (uint64_t)UINT64_MAX;
    Vett2[0] = (uint64_t)6;
    Vett1[1] = (uint64_t)1;
    Vett2[1] = (uint64_t)6;
    Vett1[2] = (uint64_t)4;
//    Vett2[2] = UINT64_MAX;
    Vett2[2] = (uint64_t)8;
 //   printf("%lu", Vett2[2]);
    Vett1[3] = (uint64_t)0;
   // Vett2[3] = (uint64_t)UINT64_MAX;
    //Vett1[4] = (uint64_t)UINT64_MAX;
    //Vett2[4] = (uint64_t)UINT64_MAX;

   Array_Clmul(Vett1, Vett2, n1, n2, Res, n3);

    return 0;
}

void Array_Clmul(uint64_t Vett1[], uint64_t Vett2[], uint32_t n1, uint32_t n2,__m256i Res[], uint32_t n3) {

    __m256i ResTemp;
    int64_t i, j;
	__m128i V1, V2;

    for(i = 0; i <  (n1 + 1 >> 1) + (n2 + 1 >> 1) - 1; i++){
       Res[i] = _mm256_set_epi64x ( (uint64_t)0, (uint64_t)0, (uint64_t)0, (uint64_t)0);
        //Res[i] = _mm256_setzero_si256;                 //setzero F
        //print__m256(Res[i]);
    }

    for(j = 0; j < (n2 >> 1); j++ ){        //Ciclo j n2/2 per difetto
        V2 = _mm_set_epi64x ( Vett2[2*j + 1] , Vett2[2*j]);

        for(i = 0; i < (n1 >> 1) ; i++ ){       //Ciclo i n1/2 per difetto
            V1 = _mm_set_epi64x ( Vett1[2*i + 1] , Vett1[2*i]);
            ResTemp = clmul( (__m128i)V1 ,(__m128i)V2);     //Clmul Temporanea
            Res[j + i] ^= ResTemp;             //Xor con Res e Temp in Res
        }
    }

    if( n1 % 2 == 1){        //n1 dispari
            V1 = _mm_set_epi64x ( 0 , Vett1[n1 - 1]);

            for(j = 0; j < (n2 >> 1); j++ ){
                V2 = _mm_set_epi64x ( Vett2[2*j+1] , Vett2[2*j]);
                ResTemp = clmul( (__m128i)V1 ,(__m128i)V2);
                Res[ j + (n1 + 1 >> 1) - 1] ^= ResTemp;
                }
    }


    if( n2 % 2 == 1){       //n2 dispari
            V2 = _mm_set_epi64x ( 0 , Vett2[n2-1]);

            for(i = 0; i < (n1 >> 1) ; i++ ){
                V1 = _mm_set_epi64x ( Vett2[2*i+1] , Vett2[2*i]);
                ResTemp = clmul( (__m128i)V1 ,(__m128i)V2);
                Res[ i + (n2 + 1 >> 1) - 1] ^= ResTemp;
            }
    }

    if( (n2 % 2 == 1) && (n1 % 2 == 1) ){      //n1 e n2 dispari
            V1 = _mm_set_epi64x ( 0 , Vett1[n1-1]);
            V2 = _mm_set_epi64x ( 0 , Vett2[n2-1]);
            ResTemp = clmul( (__m128i)V1 ,(__m128i)V2);
            Res[ (n1 + 1 >> 1) + (n2 + 1 >> 1) -1 ] ^= ResTemp;
    }

    for(j = 0; j < (n1 + 1 >> 1) + (n2 + 1 >> 1) - 1; j++ ) {
            printf("for stampa %d", j);
            print__m256(Res[j]);
    }
    return 0;
}
