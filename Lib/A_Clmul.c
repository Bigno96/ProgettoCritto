#include "A_Clmul.h"

#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>

// clmul fra num1 e num2, salvata in ris
__m256i clmul (__m128i val1, __m128i val2);
// clmul fra due Array, salvata in Res
void Array_Clmul(uint32_t n3, uint64_t Res[], uint32_t n1, uint64_t Vett1[],  uint32_t n2, uint64_t Vett2[]);
// stampa m256i
void print__m256 (__m256i num);
// stampa m128i
void print__m128 (__m128i num);

// stampa m256i
void print__m256 (__m256i num) {
    alignas(32) uint32_t v[8];
    _mm256_store_si256((__m256i*)v, num);
    printf("0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X\n", v[7], v[6], v[5], v[4], v[3], v[2], v[1], v[0]);
}

// stampa mm128i
void print__m128 (__m128i num) {
    alignas(16) uint32_t v[4];
    _mm_store_si128((__m128i*)v, num);
    printf("0x%08X 0x%08X 0x%08X 0x%08X\n", v[3], v[2], v[1], v[0]);
}

// clmul fra num1 e num2, salvata in ris
__m256i clmul (__m128i val1, __m128i val2) {
    __m256i ris = _mm256_set_epi32 (-1, -1, -1, -1, -1, -1, -1, -1);
    // scheletro m256 -> c1 : c0+c1+d1+e1 . d1+c0+d0+e0 : d0
    asm ("movdqa %[val1], %%xmm0\n\t"         // val1 in xmm0
         "movdqa %[val2], %%xmm1\n\t"         // val2 in xmm1
         "movdqa %%xmm0, %%xmm2\n\t"              // val1 copiato in xmm2
         "pclmulqdq $17, %%xmm1, %%xmm2\n\t"     // a1 * b1 in xmm2, c1:c0
         "movdqa %%xmm0, %%xmm3\n\t"
         "pclmulqdq $0, %%xmm1, %%xmm3\n\t"     // a0 * b0 in xmm3, d1:d0
         "vpsrldq $8, %%xmm0, %%xmm4\n\t"       // xmm4 contiene a1 nella metà low
         "pxor %%xmm0, %%xmm4\n\t"          //a1 xor a0 in xmm4 (metà low)
         "vpsrldq $8, %%xmm1, %%xmm5\n\t"       // xmm5 contiene b1 nella metà low
         "pxor %%xmm1, %%xmm5\n\t"          //b1 xor b0 in xmm5 (metà low)
         "pclmulqdq $0, %%xmm5, %%xmm4\n\t"     // xmm4 contiene (a0 xor a1) * (b0 xor b1), e1:e0
         "pxor %%xmm3, %%xmm4\n\t"           // in xmm4 ho d1 xor e1 : d0 xor e0
         "pxor %%xmm2, %%xmm4\n\t"          // in xmm4 ho c1 xor d1 xor e1 : c0 xor d0 xor e0
         "vpsrldq $8, %%xmm4, %%xmm5\n\t"     // shifto in low di xmm5 la parte high di xmm4
         "pxor %%xmm2, %%xmm5\n\t"           // xmm5 contiene c1 : c0+c1+d1+e1
         "pslldq $8, %%xmm4\n\t"             // shifto in high di xmm4 la parte low di xmm4
         "pxor %%xmm3, %%xmm4\n\t"            // xmm4 contiene d1+c0+d0+e0 : d0
         // xmm5 : xmm4 sono la soluzione
         "vinserti128 $0, %%xmm4, %%ymm0, %%ymm0\n\t"           //copio xmm4 in low di ymm0
         "vinserti128 $1, %%xmm5, %%ymm0, %%ymm0\n\t"           //copio xmm5 in high di ymm0
         "vmovdqa %%ymm0, %[ris]\n\t"          //ret
         : [ris] "=m" (ris)
         : [val1] "m" (val1), [val2] "m" (val2)
         );

    return ris;
}

// clmul fra due Array, salvata in Res
void Array_Clmul(uint32_t n3, uint64_t Res[], uint32_t n1, uint64_t Vett1[],  uint32_t n2, uint64_t Vett2[]) {

    int64_t i=0, j=0;
    int64_t lTmp = ((n1+1)>>1) + ((n2+1)>>1) -1;        //lunghezza (in 256) del vettore Res Temp, che raccoglie i risultati in uscita dai blocchi di clmul
    int64_t l256 = (n3+3) >> 2;                         //lunghezza (in 256) del vettore che raccoglie le somme parziali degli elementi di Res Temp
    __m256i ResTemp[lTmp], Res256[l256];
    __m256i ZERO;                                       //elemento ZERO per realizzare lo shift con permute
    __m128i V1, V2;                                     //128 bit da passare in ingresso a clmul

    ZERO = _mm256_set_epi64x ((uint64_t)0, (uint64_t)0, (uint64_t)0, (uint64_t)0);      // setto ZERO a zero

    //inizializzo Res256 e ResTemp
    for(i = 0; i < l256; i++)
    {
        Res256[i] = _mm256_set_epi64x ((uint64_t)0, (uint64_t)0, (uint64_t)0, (uint64_t)0);
    }
    for(i = 0; i < lTmp; i++)
    {
        ResTemp[i] = _mm256_set_epi64x ((uint64_t)0, (uint64_t)0, (uint64_t)0, (uint64_t)0);
    }

    for(j = 0; j < (n2 >> 1); j++)              //ciclo su Vett2, se pari lo finisco
    {
        V2 = _mm_set_epi64x (Vett2[2*j + 1], Vett2[2*j]);       //blocco la prima "coppia" di 64 di Vett2

        for(i = 0; i < (n1 >> 1) ; i++)                         //ciclo su Vett1, lo finisco se è pari
        {
            V1 = _mm_set_epi64x (Vett1[2*i + 1], Vett1[2*i]);   //moltiplico V2 per tutti i blocchi di Vett1
            ResTemp[j+i] ^= clmul((__m128i)V1, (__m128i)V2);    //salvo in ResTemp
        }

        if(n1 % 2 == 1)                                         //se Vett1 dispari
        {
            V1 = _mm_set_epi64x (0, Vett1[n1-1]);               //metto l'ultimo blocco di Vett1 in v1, inserendo 0 nella parte "più significativa"
            ResTemp[j+(n1>>1)] ^= clmul((__m128i)V1, (__m128i)V2);          //salvo in ResTemp l'ultima moltiplicazione
        }
    }

    if(n2 % 2 == 1)                             //se Vett2 dispari
    {
        V2 = _mm_set_epi64x (0, Vett2[n2-1]);                   //blocco l'ultima coppia di Vett2

        for(i = 0; i < (n1 >> 1) ; i++)                         //ciclo su Vett1, se pari lo finisco
        {
            V1 = _mm_set_epi64x (Vett1[2*i + 1], Vett1[2*i]);       //moltiplico V2 per tutti i blocchi di Vett1
            ResTemp[(n2>>1)+i] ^= clmul((__m128i)V1, (__m128i)V2);  //salvo in ResTemp
        }

        if(n1 % 2 == 1)                                         //se Vett1 dispari
        {
            V1 = _mm_set_epi64x (0, Vett1[n1-1]);               //metto l'ultimo blocco di Vett1 in v1, inserendo 0 nella parte "più significativa"
            ResTemp[lTmp-1] ^= clmul((__m128i)V1, (__m128i)V2);     //salvo in ResTemp l'ultima moltiplicazione
        }
    }

    Res256[0] = ResTemp[0] ^ _mm256_permute2x128_si256(ZERO, ResTemp[1], 33);       //shifta il 128 low di ResTemp[1] nel 128 high e fa l'or con ResTemp[0]
    for (i = 1; i < l256 - 1; i++)                              //ciclo escludendo prima e ultima cifra di Res256, trattate prima e dopo
    {
        // res = i pari di resTemp  or  la parte high >> in low del resTemp precedente  or  la parte low >> high del resTemp successivo
        Res256[i] = ResTemp[2*i] ^ _mm256_permute2x128_si256(ResTemp[2*i-1], ZERO, 33) ^ _mm256_permute2x128_si256(ZERO, ResTemp[2*i+1], 33);
    }
    Res256[l256-1] = ResTemp[lTmp-1] ^ _mm256_permute2x128_si256(ResTemp[lTmp-2], ZERO, 33);       // shifta il 128 high di ResTemp finale nel 128 low e fa l'or con ResTemp finale

    alignas (32) uint64_t v[4];
    //converto Res256 da elementi a 256 bit a elementi Digit 64 bit
    for (j = 0; j < l256; j++)      // ciclo su Res256
    {
        _mm256_store_si256((__m256i*)v, Res256[j]);     //salvo in memoria Res256[j] e lo associo al vettore v
  
        for (i = 0; i < 4; i++)         //4 * 64 = 256
        {
            Res[(j*4)+i] = v[i];        //ogni elemento a 64bit di Res256[j] viene salvato in Res con l'offset adeguato
        }
    }
}
