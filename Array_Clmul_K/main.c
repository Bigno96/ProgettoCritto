#include "A_Clmul.h"

#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>
#include <inttypes.h>

void Add(uint32_t n3, uint64_t Res[], uint32_t n1, uint64_t Vett1[], uint32_t n2, uint64_t Vett2[]);
void Print_Vett64(uint32_t n, uint64_t Vett[]);
void Print_Vett256(uint32_t n, __m256i Vett[]);
void ACK(uint32_t n3, uint64_t Res[], uint32_t n1, uint64_t Vett1[], uint32_t n2, uint64_t Vett2[]);

int main(int argc, char *argv[])
{
    uint32_t n1 = 2, n2 = 3;
    uint32_t n3 = n1 + n2;
    uint32_t i = 0;
    uint64_t Vett1[n1], Vett2[n2], Res[n3];

    Vett1[0] = UINT64_MAX;
    Vett1[1] = UINT64_MAX;

    Vett2[0] = UINT64_MAX;
    Vett2[1] = 0;
    Vett2[2] = UINT64_MAX;

    for (i = 0; i < n3; i++)
    {
        Res[i] = 0;
    }

    ACK(n3, Res, n1, Vett1, n2, Vett2);

    return 0;
}

void Add(uint32_t n3, uint64_t Res[], uint32_t n1, uint64_t Vett1[], uint32_t n2, uint64_t Vett2[])     //Add
{
    uint32_t i;
    for(i = 0; i < n3; i++)
    {
        Res[i] = Vett1[i] ^ Vett2[i];
    }

}

void Print_Vett64(uint32_t n, uint64_t Vett[])
{
    uint32_t i;
    for (i = 0; i < n; i++)
    {
        printf("[%u] 0x%016lX\n", i, Vett[i]);
    }
    printf("\n");

}

void Print_Vett256(uint32_t n, __m256i Vett[])
{
    uint32_t i;
    for (i = 0; i < n; i++)
    {
        printf("[%u] ", i);
        print__m256(Vett[i]);
    }
    printf("\n");
}

void ACK(uint32_t n3, uint64_t Res[], uint32_t n1, uint64_t Vett1[], uint32_t n2, uint64_t Vett2[])     //moltiplicazione da dx a sx (0 -> n)
{
    int32_t a = (n1+1) >> 1;                                //lunghezza dei vettori A0 e A1
    int32_t b = (n2+1) >> 1;                                //lunghezza dei vettori B0 e B1
    uint32_t l = ((a+1) >> 1) + ((b+1) >> 1) - 1;           //lunghezza dei vettori risultato di Array_Clmul
    int32_t i = 0;
    uint64_t A0[a], A1[a], B0[b], B1[b];    //A0 parte meno significativa, riempita con il bit medio. A1 parte più significativa, riempita con uno zero.
    uint64_t SumA[a], SumB[b];              //temporanei per gli xor A0 + A1, B0 + B1
    __m256i C[l], D[l], E[l];               //temporanei per l'uscita da Array Clmul

    // A0 <= da Vett1[0] a Vett[n1/2] (se dispari n1+1)

    for(i = 0; i < (n1) >> 1; i++)          //riempio A0 e A1, ignorando il bit "in mezzo" se dispari
    {
        A0[i] = Vett1[i];
        A1[i] = Vett1[a+i];
    }

    if(n1 % 2 == (uint32_t)1)               //se n1 dispari
    {
        A0[a-1] = Vett1[a-1];               //bit finale di A0
        A1[a-1] = (uint64_t) 0;             //bit finale di A1
    }

    for(i = 0; i < (n2) >> 1; i++)          //riempio B0 e B1, ignorando il bit "in mezzo" se dispari
    {
        B0[i] = Vett2[i];
        B1[i] = Vett2[b+i];
    }

    if(n2 % 2 == (uint32_t)1)               //se n2 dispari
    {
        B0[b-1] = Vett2[b-1];               //bit finale di B0
        B1[b-1] = (uint64_t) 0;             //bit finale di B1
    }

    Array_Clmul(l, C, a, A0, b, B0);        //A0 * B0 ottengo C1, C0
    Array_Clmul(l, D, a, A1, b, B1);        //A1 * B1 ottengo D1, D0
    printf("Stamp C0 : C1\n");
    Print_Vett256(l, C);
    printf("Stamp D0 : D1\n");
    Print_Vett256(l, D);

    Add(a, SumA, a, A0, a, A1);             //A0 + A1
    Add(b, SumB, b, B0, b, B1);             //B0 + B1

    Array_Clmul(l, E, a, SumA, b, SumB);    //(A0 + A1) * (B0 + B1)
    printf("Stamp E0 : E1\n");
    Print_Vett256(l, E);

}
