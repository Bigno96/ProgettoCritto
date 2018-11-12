CC = gcc
CFLAGS = -mavx -mavx2 -mpclmul -O2 
SRC = ack.c
OBJ = ack.o

ack: $(OBJ) $(SRC)
	$(CC) -c $(SRC) $(CFLAGS)
	ar rcs ack.a $(OBJ)
	

