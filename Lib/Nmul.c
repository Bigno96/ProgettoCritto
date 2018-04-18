#include "Nmul.h"

#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>

#define N 256

// restituisce int a 64bit a partire da 64 bit di un vettore, DISCENDENTE
uint64_t parse_num_64 (uint64_t vet[], uint32_t pos);
// riempie 64 bit di un vettore con un int a 64bit, dalla pos specificata, DISCENDENTE
void parse_vet_64 (uint64_t num, uint64_t vet[], uint32_t pos);
// clmul fra num1 e num2, salvata in ris
void normal_mul (__m128i *num1, __m128i *num2, __m256i *ris);

// restituisce int a 64bit a partire da 64 bit di un vettore, DISCENDENTE
uint64_t parse_num_64 (uint64_t vet[], uint32_t pos) {
    int32_t e=0, i=0;
    uint64_t num=0;
    // ciclo che parte dalla posizione e finisce dopo 63 elementi
    for (i=pos, e=62; i>pos-63; e--, i--) {
        if (vet[i]==1)
            num|=(uint64_t)2<<e;        // se vet[i] = 1, aggiungo 2^e al numero, altrimenti nulla
    }
    // controllo l'ultimo elemento, che corrisponde a 2^0
    if (vet[pos-63]==1)
        num|=(uint64_t)1;
    return num;
}

// riempie 64bit di un vettore con un int a 64 bit, dalla pos specificata, DISCENDENTE
void parse_vet_64 (uint64_t num, uint64_t vet[], uint32_t pos) {
    int32_t e=0, i=0;
    uint64_t resto=0;
    // ciclo che parte dalla cifra 2^63 e finisce a 2^1
    for (i=pos, e=62; i>pos-63; e--, i--) {
        resto = num%((uint64_t)2<<e);     // ottengo il resto della divisione fra num e 2^e
        if (resto==num) {
            vet[i]=0;
        } else {
            vet[i]=1;
        }
        num = resto;                // ad ogni iterazione, setto num=resto
    }
    // controllo 2^0
    if (num==1)
        vet[pos-63]=1;
}

// clmul fra num1 e num2, salvata in ris
void normal_mul (__m128i *num1, __m128i *num2, __m256i *ris) {
    int32_t i=0, j=0;
    uint64_t vet1[N], vet2[N];
    uint64_t res[2*N];   // arrivo a 256 bit perch� anche se la somma � lunga 255, __m256 � a 256 bit. Setter� res[255]=0.

    // inizializzo
    for (i=0; i<N; i++) {
        vet1[i]=0;
    }
    for (i=0; i<N; i++) {
        vet2[i]=0;
    }
    for (i=0; i<2*N-1; i++) {
        res[i]=0;
    }

    alignas(16) uint64_t v[2];       // creo vettore v di 2 elementi, allineo a 16 per necessit� di mm_store

    _mm_store_si128((__m128i*)v, *num1);    // ora v in memoria contiene num1, quindi i 2 int64 di v sono i 2 int64 di num1
    // setto vet 1-> if num1 = (1,2), vet1 = (2,1)
    parse_vet_64(v[0], vet1, 63);
    parse_vet_64(v[1], vet1, 127);

    _mm_store_si128((__m128i*)v, *num2);    // ora v in memoria contiene num2, quindi i 2 int64 di v sono i 4 int64 di num2
    // setto vet 2-> if num2 = (1,2), vet2 = (2,1)
    parse_vet_64(v[0], vet2, 63);
    parse_vet_64(v[1], vet2, 127);

    for (i=0; i<N; i++) {           // scorro vet 2
        if (vet2[i]==1) {               // se vet2 = 0, non faccio nulla, altrimenti riporto vet1 uguale
            for (j=0; j<N; j++) {        // sommo vet1 a ris partendo da ris[i+j] (come se shiftassi vet1)
                res[i+j] = vet1[j] ^ res[i+j];
            }
        }
    }
    res[255]=0;     // setto ultimo bit a 0 per le dim di __m256
    // setto ris
    *ris = _mm256_set_epi64x(parse_num_64(res, 255), parse_num_64(res, 191), parse_num_64(res, 127), parse_num_64(res, 63));
}

