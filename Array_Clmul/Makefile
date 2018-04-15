Clmul.o: Clmul.c Clmul.h
	gcc -Wall -pedantic -c -mavx Clmul.c
 
Clmul.a: Clmul.o
	ar r Clmul.a Clmul.o

main.o: main.c Clmul.h
	gcc -Wall -pedantic -c -mavx  main.c

