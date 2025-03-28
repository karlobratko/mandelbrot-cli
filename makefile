TARGET = mandelbrot

CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O3 -D_POSIX_C_SOURCE=200809L -march=native
LDFLAGS = -lm -lc

SRC = mandelbrot.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) *.ppm

.PHONY: all clean
