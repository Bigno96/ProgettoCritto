A_Clmul.a: A_Clmul.o
	ar r A_Clmul.a A_Clmul.o

A_Clmul.o: A_Clmul.c A_Clmul.h
	gcc -Wall -pedantic -c -mavx A_Clmul.c
 
Clmul.a: Clmul.o
	ar r Clmul.a Clmul.o

Clmul.o: Clmul.c Clmul.h
	gcc -Wall -pedantic -c -mavx Clmul.c


