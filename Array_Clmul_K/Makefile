A_Clmul.o: A_Clmul.c A_Clmul.h
	gcc -Wall -pedantic -c -mavx A_Clmul.c
 
A_Clmul.a: A_Clmul.o
	ar r A_Clmul.a A_Clmul.o

main.o: main.c A_Clmul.h
	gcc -Wall -pedantic -c -mavx  main.c

