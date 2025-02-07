all: server client

server:
	$(CC) $(CFLAGS) server.c -o server

client:
	$(CC) $(CFLAGS) client.c -o client

# %.o: %.c
# 	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f server client

.PHONY: all clean
