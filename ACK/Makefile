CC=gcc
C=main.c
O=main.o
CFLAGS= -mavx -mavx2 -std=gnu11 -I -O2.

make: $(C)
	$(CC) $(CFLAGS) -c $(C)
	ar rs ack.a $(O)

