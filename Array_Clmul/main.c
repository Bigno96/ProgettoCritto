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
    n1 = 9;
    n2 = 5
         ;
    n3 = n1 + n2;

    uint64_t Vett1[N], Vett2[N];
    uint64_t Res[N];

    Vett1[0] = (uint64_t)UINT64_MAX;
    Vett1[1] = (uint64_t)UINT64_MAX;
    Vett1[2] = (uint64_t)UINT64_MAX;
    Vett1[3] = (uint64_t)UINT64_MAX;
    Vett1[4] = (uint64_t)UINT64_MAX;
    Vett1[5] = (uint64_t)UINT64_MAX;
    Vett1[6] = (uint64_t)UINT64_MAX;
    Vett1[7] = (uint64_t)UINT64_MAX;
    Vett1[8] = (uint64_t)UINT64_MAX;


    Vett2[0] = (uint64_t)0;
    Vett2[1] = (uint64_t)0;
    Vett2[2] = (uint64_t)0;
    Vett2[3] = (uint64_t)0;
    Vett2[4] = (uint64_t)1;

    Array_Clmul(n3, Res, n1, Vett1, n2, Vett2);

    return 0;

}

void Array_Clmul(uint32_t n3, uint64_t Res[], uint32_t n1, uint64_t Vett1[], uint32_t n2, uint64_t Vett2[])
{
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

    printf("lTmp = %ld\nl256 = %ld\n", lTmp, l256);

    for(i=0; i < lTmp; i++)
    {
        printf("\nResTemp: \n");
        print__m256(ResTemp[i]);
    }

    Res256[0] = ResTemp[0] ^ _mm256_permute2x128_si256(ZERO, ResTemp[1], 33);       //shifta il 128 low di ResTemp[1] nel 128 high e fa l'or con ResTemp[0]
    for (i = 1; i < l256 - 1; i++)                              //ciclo escludendo prima e ultima cifra di Res256, trattate prima e dopo
    {
        // res = i pari di resTemp  or  la parte high >> in low del resTemp precedente  or  la parte low >> high del resTemp successivo
        Res256[i] = ResTemp[2*i] ^ _mm256_permute2x128_si256(ResTemp[2*i-1], ZERO, 33) ^ _mm256_permute2x128_si256(ZERO, ResTemp[2*i+1], 33);
    }
    Res256[l256-1] = ResTemp[lTmp-1] ^ _mm256_permute2x128_si256(ResTemp[lTmp-2], ZERO, 33);       // shifta il 128 high di ResTemp finale nel 128 low e fa l'or con ResTemp finale

    alignas (32) uint64_t v[4];
    printf("\nRes256: \n");         //converto Res256 da elementi a 256 bit a elementi Digit 64 bit
    for (j = 0; j < l256; j++)      // ciclo su Res256
    {
        _mm256_store_si256((__m256i*)v, Res256[j]);     //salvo in memoria Res256[j] e lo associo al vettore v
        print__m256(Res256[j]);
        for (i = 0; i < 4; i++)         //4 * 64 = 256
        {
            Res[(j*4)+i] = v[i];        //ogni elemento a 64bit di Res256[j] viene salvato in Res con l'offset adeguato
            printf("[%ld] 0x%016lX\n", i, v[i]);
            printf("\n\n");
        }
    }

    for(i=n3-1; i >=0; i--)
    {
        printf("\nRes: \n");
        printf("[%ld] 0x%016lX\n", i, Res[i]);
    }

}
