#include <stdint.h>
#define BYTES_PER_PIXEL 3
#define INIT_CLIENT_FLAG 1
#define DATA_IN_FLAG 2
// for the server side
int tcp_recv_setup(int portNum);
int tcp_listen(int server_socket, int back_log);

// for the client side
int tcp_send_setup(char *host_name, char *port);

typedef struct Header {
    uint8_t flag;
    uint32_t size;
    uint32_t frame_num;
} __attribute__((packed)) Header;
typedef struct Pixel {
   char r;
   char g;
   char b;
} Pixel;
