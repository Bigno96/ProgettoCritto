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

    return;
}

void Print_Vett64(uint32_t n, uint64_t Vett[])
{
    uint32_t i;
    for (i = 0; i < n; i++)
    {
        printf("[%u] %016lX\n", i, Vett[i]);
    }
    printf("\n");
    return;

}

void ACK(uint32_t n3, uint64_t Res[], uint32_t n1, uint64_t Vett1[], uint32_t n2, uint64_t Vett2[])     //moltiplicazione da dx a sx (0 -> n)
{
    uint32_t l = ((n1+1) >> 1) + ((n2+1) >> 1) - 1;         //lunghezza dei vettori risultato di Array_Clmul
    int32_t a = (n1+1) >> 1;                               //lunghezza dei vettori A0 e A1
    int32_t b = (n2+1) >> 1;                               //lunghezza dei vettori B0 e B1
    int32_t i = 0;
    uint64_t A0[a], A1[a], B0[b], B1[b];   //A0 parte meno significativa, riempita con il bit medio. A1 parte più significativa, riempita con uno zero.
    __m256i C[l], D[l], E[l];

    // A0 <= da Vett1[0] a Vett[n1/2] (se dispari n1+1)

    for(i = 0; i < (n1) >> 1; i++)          //riempio A0 e A1, ignorando il bit "in mezzo" se dispari
    {
        A0[i] = Vett1[i];
        A1[i] = Vett1[a+i];
    }

    if(n1 % 2 == (uint32_t)1)                  //se n1 dispari
    {
        A0[a-1] = Vett1[a-1];       //bit finale di A0
        A1[a-1] = (uint64_t) 0;             //bit finale di A1
    }

    for(i = 0; i < (n2) >> 1; i++)          //riempio B0 e B1, ignorando il bit "in mezzo" se dispari
    {
        B0[i] = Vett2[i];
        B1[i] = Vett2[b+i];
    }

    if(n2 % 2 == (uint32_t)1)                  //se n2 dispari
    {
        B0[b-1] = Vett2[b-1];       //bit finale di B0
        B1[b-1] = (uint64_t) 0;             //bit finale di B1
    }

    printf("Vettore A0:\n");
    Print_Vett64(a, A0);
    printf("Vettore A1:\n");
    Print_Vett64(a, A1);
    printf("Vettore B0:\n");
    Print_Vett64(b, B0);
    printf("Vettore B1:\n");
    Print_Vett64(b, B1);




}
