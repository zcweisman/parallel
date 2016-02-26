/******************************************************************************
 * tcp_client.c
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <ctype.h>

#include "common.h"

#define BUF_SIZE 2000
#define MAX_MESSAGE_LENGTH 1000
#define HANDLE_SIZE 101
#define MESSAGE_SIZE 2000
#define CMD_OFFSET 3
#define CMD_SIZE 4

void HandleData(char *buf);

int init_client(int seq, char *handle, char *send_buf);

int get_input(char *in_buf);

int handle_cmd(char cmd, char *send_buf, char *handle, Header *header, char *in_buf, int in_len, int *length, int *print);

int build_message(char *send_buf, char *handle, Header *header, char *in_buf, int *length, int *print);

int build_broadcast(char *send_buf, char *handle, Header *header, char *in_buf, int *length);

void build_quit(char *send_buf, Header *header);

int handle_server_data(char *in_buf, char *handle, int in_len, int socket_num);

void handle_send_error(char *in_buf);

void handle_incoming_message(char *in_buf);

void handle_incoming_broadcast( char *in_buf);

void handle_incoming_list(char *in_buf, int in_len, int socket_num);

void Prompt(int *print, int *err);

void Send(int *code, int *sent, int *seq, int *socket_num, char *send_buf, int *send_len);

void greyscale(uint8_t *data, int size);

int main(int argc, char * argv[])
{
   int socket_num;         //socket descriptor
   char send_buf[BUF_SIZE];         //data buffer
   int send_len= 0;        //amount of data to send
   char in_buf[BUF_SIZE];         //data buffer
   char cmd_buf[CMD_SIZE];
   int in_len = 0;
   int sent= 0;            //actual amount of data sent
   u_int32_t seq = 0;
   Header header;
   int code = 0;
   int err = 0;
   int result = 0;
   int print = 1;
   fd_set readSet;
    
   /* check command line arguments  */
   if(argc!= 4)
   {
      printf("usage: %s host-name port-number handle\n", argv[0]);
      exit(1);
   }
   send_len = init_client(seq++, argv[3], send_buf); 
   /* set up the socket for TCP transmission  */
   socket_num= tcp_send_setup(argv[1], argv[2]);
   send(socket_num, send_buf, send_len, 0);
   /* get the data and send it */ 
   do {
      FD_ZERO(&readSet);
      FD_SET(STDIN_FILENO, &readSet);
      FD_SET(socket_num, &readSet);
      Prompt(&print, &err); 
      result = select(socket_num + 1, &readSet, NULL, NULL, NULL); 
      if(result < 0) {
         perror("select call");
         exit(-1);
      }
      if (FD_ISSET(STDIN_FILENO, &readSet)) {
         in_len = read(STDIN_FILENO, in_buf, BUF_SIZE);
         sscanf(in_buf, " %3s", cmd_buf);
         if (cmd_buf[2] == '\0' && cmd_buf[0] == '%') {
            code = handle_cmd(cmd_buf[1], send_buf, argv[3], &header, in_buf + strlen(cmd_buf), in_len, &send_len, &print);
            Send(&code, &sent, &seq, &socket_num, send_buf, &send_len); 
         }
         else {
            print = 1;
            printf("Invalid command\n");
         }
      }
      if(FD_ISSET(socket_num, &readSet)) {
         in_len = recv(socket_num, in_buf, BUF_SIZE, 0);
         if (in_len <= 0) {
            err = 2;
            printf("\nServer Terminated");
         }
         else {
            err = handle_server_data(in_buf, argv[3], in_len, socket_num);
            SetFlags(&err, &print); 
         }
      }
    } while(err != 2);
    close(socket_num);
    return 0; 
}
void Send(int *code, int *sent, int *seq, int *socket_num, char *send_buf, int *send_len) {
   if (!*code) {
      *sent = send(*socket_num, send_buf, *send_len, 0);
      (*seq)++;
      if(*sent < 0)
      {
         perror("send call");
         exit(-1);
      }
   }
}
void Prompt(int *print, int *err) {
   if (*print) {
      printf("$: ");
      fflush(stdout);
      *err = 0;
   }
}

int handle_server_data(char *in_buf, char *handle, int in_len, int socket_num) {
   Header *header = (Header *)in_buf;
   int code = 0; 
   switch (header->flag) {
      case 2:
         code = 3;
         break;
      case 3:
         printf("\nHandle already in use: %s\n", handle);
         code = 1;
         break;
      case 4:
         handle_incoming_broadcast(in_buf + sizeof(Header));
         break;
      case 5:
         handle_incoming_message(in_buf + sizeof(Header));
         break;
      case 6:
         code = 3;
         break;
      case 7:
         handle_send_error(in_buf + sizeof(Header));
         break;
      case 9:
         code = 2;
         break;
      case 11:
         handle_incoming_list(in_buf + sizeof(Header), in_len, socket_num);
         break;
      default:
         code = 3;
         break;
   }
   return code;
}

void handle_incoming_list(char *in_buf, int in_len, int socket_num) {
   u_int32_t numHandles = ntohl(*((u_int32_t *)in_buf));
   u_int32_t handleCount = 0;
   u_int32_t buf_left = in_len - sizeof(Header);
   char curHandleSize = 0;
   char needRestOfHandle = 0;
   char sizeOrHandle = 0;
   char handlebuf[HANDLE_SIZE];
   char handleBufOffset = 0; 
   char *temp = in_buf;
   in_buf += sizeof(u_int32_t);
   buf_left -= sizeof(u_int32_t);

   printf("Clients Currently Connected:\n");
   while (handleCount < numHandles) {
      if (!buf_left) {
         buf_left = recv(socket_num, temp, BUF_SIZE, 0);
         in_buf = temp;
      }
      else if(sizeOrHandle == 0) {
         curHandleSize = *in_buf;
         in_buf++;
         buf_left--;
         sizeOrHandle = 1;
      }
      else if (curHandleSize > buf_left) {
         memcpy(handlebuf, in_buf, buf_left);
         needRestOfHandle = 1;
         handleBufOffset = buf_left; 
         handlebuf[buf_left] = '\0';
         buf_left = 0;
      }
      else if (needRestOfHandle == 1) {
         memcpy(handlebuf + handleBufOffset, in_buf, curHandleSize - handleBufOffset);
         needRestOfHandle = 0;
         handlebuf[curHandleSize] = '\0';
         handleCount++;
         in_buf += (curHandleSize - handleBufOffset);
         buf_left -= (curHandleSize - handleBufOffset);
         handleBufOffset = 0;
         sizeOrHandle = 0;
         printf("%s\n", handlebuf);
      }
      else if(sizeOrHandle == 1) {
         memcpy(handlebuf, in_buf, curHandleSize);
         handlebuf[curHandleSize] = '\0';
         in_buf += curHandleSize;
         handleCount++;
         buf_left -= curHandleSize;
         sizeOrHandle = 0;
         curHandleSize = 0;
         printf("%s\n", handlebuf);
      }
   }
}

void handle_incoming_broadcast(char *in_buf) {
   char sendHandleLen = *in_buf++;
   char sendHandle[HANDLE_SIZE];

   memcpy(sendHandle, in_buf, sendHandleLen);
   in_buf += sendHandleLen;
   sendHandle[sendHandleLen] = '\0';
   printf("\n%s: %s\n", sendHandle, in_buf);
}

void handle_incoming_message(char *in_buf) {
   char destHandleLen = *in_buf;
   char destHandle[HANDLE_SIZE];
   char sendHandleLen = 0;
   char sendHandle[HANDLE_SIZE];

   in_buf++;
   memcpy(destHandle, in_buf, destHandleLen);
   in_buf += destHandleLen;
   sendHandleLen = *in_buf++;
   memcpy(sendHandle, in_buf, sendHandleLen);
   sendHandle[sendHandleLen] = '\0';
   in_buf += sendHandleLen;
   printf("%s: %s\n", sendHandle, in_buf);

}

void handle_send_error(char *in_buf) {
   char len = *in_buf;
   char handle[HANDLE_SIZE];
   in_buf++;
   memcpy(handle, in_buf, len);
   handle[len] = '\0';
   printf("Client with handle %s does not exist.\n", handle);
}

int handle_cmd(char cmd, char *send_buf, char *handle, Header *header, char *in_buf, int in_len, int *length, int *print) {
   int code = 0;

   switch (toupper(cmd)) {
      case 'M':
         code = build_message(send_buf, handle, header, in_buf, length, print);
         break;
      case 'B':
         *print = 1;
         code = build_broadcast(send_buf, handle, header, in_buf, length);
         break;
      case 'L':
         *print = 0;
         header->flag = 10;
         memcpy(send_buf, header, sizeof(Header));
         *length = sizeof(Header);
         break;
      case 'E':
         *print = 0;
         build_quit(send_buf, header);
         break;
      default:
         *print = 1;
         printf("Invalid command\n");
         code = 1;
         break;
   }
   return code;
}

void build_quit(char *send_buf, Header *header) {
   header->flag = 8;
   memcpy(send_buf, header, sizeof(Header));
}

int build_broadcast(char *send_buf, char *handle, Header *header, char *in_buf, int *length) {
   char sendHandleLen = strlen(handle);
   int messageLen = 0;
   char message_buf[MESSAGE_SIZE];
   int code = 0;

   sscanf(in_buf, " %[^\n]%*c", message_buf);
   header->flag = 4;
   memcpy(send_buf, header, sizeof(Header));
   send_buf += sizeof(Header);
   memcpy(send_buf, &sendHandleLen, sizeof(char));
   send_buf++;
   memcpy(send_buf, handle, sendHandleLen);
   send_buf += sendHandleLen;
   messageLen = strlen(message_buf);
   if (strlen(message_buf) > 1000) {
      printf("Error, message too long, message length is: %d\n", strlen(message_buf));
      code = 1;
   }
   else {
      *length = sizeof(Header) + 1 + messageLen + 1 + sendHandleLen;
      memcpy(send_buf, message_buf, messageLen + 1);
   }
   return code;
}

int build_message(char *send_buf, char *handle, Header *header, char *in_buf, int *length, int *print) {
   char destHandleLen;
   char sendHandleLen;
   char handle_buf[HANDLE_SIZE];
   char message_buf[MESSAGE_SIZE];
   char result = 0;
   int code = 0;

   result = sscanf(in_buf, " %100s %[^\n]%*c", handle_buf, message_buf);
   if (result < 1) {
      printf("Error, no handle given\n");
      code = 1;
   }
   else {
      if(result == 1) {
         message_buf[0] = '\0';
      }
      header->flag = 5;
      memcpy(send_buf, header, sizeof(Header));
      send_buf += sizeof(Header);
      destHandleLen = strlen(handle_buf);
      memcpy(send_buf, &destHandleLen, sizeof(char));
      send_buf++;
      memcpy(send_buf, handle_buf, destHandleLen);
      send_buf += destHandleLen;

      sendHandleLen = strlen(handle);
      memcpy(send_buf, &sendHandleLen, sizeof(char));
      send_buf++;
      memcpy(send_buf, handle, sendHandleLen);
      send_buf += sendHandleLen;
      if (strlen(message_buf) > 1000) {
         printf("Error, message too long, message length is: %d\n", strlen(message_buf));
         code = 1;
         *print = 1;
      }
      else {
         *print = 0;
         *length = sizeof(Header) + 2 + destHandleLen + sendHandleLen + strlen(message_buf) + 1;
         memcpy(send_buf, message_buf, strlen(message_buf) + 1);
      }
   }
   return code;
}

int init_client(int seq, char *handle, char *send_buf) {
   Header header;
   header.flag = 1;
   char handleLen = strlen(handle);
   memcpy(send_buf, &header, sizeof(Header));
   send_buf += sizeof(Header);
   memcpy(send_buf, &handleLen, sizeof(char));
   send_buf++;
   memcpy(send_buf, handle, handleLen);
   return sizeof(Header) + 1 + handleLen;
}



int tcp_send_setup(char *host_name, char *port)
{
    int socket_num;
    struct sockaddr_in remote;       // socket address for remote side
    struct hostent *hp;              // address of remote host

    // create the socket
    if ((socket_num = socket(AF_INET, SOCK_STREAM, 0)) < 0)
	{
	    perror("socket call");
	    exit(-1);
	}
    

    // designate the addressing family
    remote.sin_family= AF_INET;

    // get the address of the remote host and store
    if ((hp = gethostbyname(host_name)) == NULL)
	{
	  printf("Error getting hostname: %s\n", host_name);
	  exit(-1);
	}
    
    memcpy((char*)&remote.sin_addr, (char*)hp->h_addr, hp->h_length);

    // get the port used on the remote side and store
    remote.sin_port= htons(atoi(port));

    if(connect(socket_num, (struct sockaddr*)&remote, sizeof(struct sockaddr_in)) < 0)
    {
	perror("connect call");
	exit(-1);
    }

    return socket_num;
}

