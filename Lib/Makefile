A_Clmul.a: A_Clmul.o
	ar r A_Clmul.a A_Clmul.o

A_Clmul.o: A_Clmul.c A_Clmul.h
	gcc -Wall -pedantic -c -mavx -mavx2 A_Clmul.c
 
Clmul.a: Clmul.o
	ar r Clmul.a Clmul.o

Clmul.o: Clmul.c Clmul.h
	gcc -Wall -pedantic -c -mavx -mavx2 Clmul.c
 
Nmul.a: Nmul.o
	ar r Nmul.a Nmul.o

Nmul.o: Nmul.c Nmul.h
	gcc -Wall -pedantic -c -mavx -mavx2 Nmul.c

