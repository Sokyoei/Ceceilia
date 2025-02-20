all: server client epoll_server

server:
	$(CC) $(CFLAGS) server.c -o server

client:
	$(CC) $(CFLAGS) client.c -o client

epoll_server:
	$(CC) $(CFLAGS) epoll_server.c -o epoll_server

# %.o: %.c
# 	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f server client epoll_server

.PHONY: all clean
