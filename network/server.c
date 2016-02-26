#include <GL/glew.h>
#include <GL/glut.h>
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

#include "common.h"
#include "lodepng.h"

#define BUF_SIZE 5000
#define HANDLE_SIZE 101


typedef struct Client {
   char *handle;
   int socket;
   struct Client *next;
} Client;

void HandleData(char *buf, Client **clientTable, int client_socket, int message_len);

void HandleCommand(char *buf);

Client *traverse_client_table(Client *clientTable);

void InitSet(fd_set *set, Client *clientTable, int server_socket);

int SelectSocket(fd_set *set, Client **clientTable, int server_socket, char *buf);

void SendNewClientResponse(int client_socket, Header *header, char flag);

int CheckHandle(char *handle, Client *clientTable);

void NewClient(char *buf, Client **clientTable, int client_socket, Header *header);

void SendExit(Header *header, int client_socket);

void RemoveClient(Client **clientTable, int client_socket);

void Prompt();

bool initGL();

bool initGL() {
	glewInit();
	if (! glewIsSupported
		(
		"GL_VERSION_4_1 "
		"GL_ARB_pixel_buffer_object "
		"GL_EXT_framebuffer_object "
		))
	{
			fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
			fflush(stderr);
			//return CUTFalse;
            return false;
	}

	// init openGL state
	glClearColor(0, 0, 0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// view-port
	glViewport(0, 0, 800, 800);

	return true;
}

int main(int argc, char *argv[])
{
    int server_socket= 0;   //socket descriptor for the server socket
    int client_socket= 0;   //socket descriptor for the client socket
    char buf[BUF_SIZE];              //buffer for receiving from client
    int ready_socket = 0;     //length of the received messagei
    int message_len = 0;
    int new_client;
    int frame_num = 0;
    Client *clientTable = NULL;
    fd_set readSet;
    char *inFilename = NULL;
    Pixel *image;

    if (argc > 1) {
        server_socket = tcp_recv_setup(atoi(argv[1]));
    }
    else { 
        server_socket = tcp_recv_setup(0);
    }
    printf("sockaddr: %d sockaddr_in %d\n", sizeof(struct sockaddr), sizeof(struct sockaddr_in));
    initGL();
    while(1) {
       InitSet(&readSet, clientTable, server_socket);
       Prompt();
       ready_socket = SelectSocket(&readSet, &clientTable, server_socket, buf);
       if (ready_socket == STDIN_FILENO) {
            printf("Command\n"); 
            message_len = read(STDIN_FILENO, buf, BUF_SIZE);
            HandleCommand(buf);              
       }
       else if ((message_len =  recv(ready_socket, buf, BUF_SIZE, 0)) < 0)
       {
           perror("recv call");
           ready_socket = -1;
       }
       if (ready_socket >= 0 && message_len) {
            HandleData(buf, &clientTable, ready_socket, message_len);
       }
       else {
            close(ready_socket);
            RemoveClient(&clientTable, ready_socket);
       }
       printf("Data received, length: %d\n", message_len);
    } 
    /* close the sockets */
    close(server_socket);
    close(client_socket);
}

void HandleCommand(char *buf) {
    switch (buf[0]) {
        case 'r':
            printf("Running RayTrace!\n");
            //Send flag 2 to start ray trace.        
            break;
        default:
            printf("Invalid Command.\n");
            break;
    }       
}

void Prompt() {
    printf("$: ");
    fflush(stdout);
}

int SelectSocket(fd_set *set, Client **clientTable, int server_socket, char *buf) {
   int retVal = -1;
   int max_socket = -1;
   int new_client_socket = 0;
   int message_len = 0;
   int res;
   Client *client = *clientTable;
   
   if (!client) {
      max_socket = server_socket;
   }
   while (client) {
      if (client->socket > max_socket) {
         max_socket = client->socket;
      }
      client = client->next;
   }
   res = select(max_socket + 1, set, NULL, NULL, NULL);
   if (FD_ISSET(STDIN_FILENO, set)) {
       return STDIN_FILENO;
   }
   else if (FD_ISSET(server_socket, set)) { 
       new_client_socket = tcp_listen(server_socket, 5);
       return new_client_socket;
   }
   else {
      client = *clientTable;
      while (client) {
         if (FD_ISSET(client->socket, set)) {
            return client->socket; 
         }
         client = client->next;
      }
   }
   return -1;
}

void InitSet(fd_set *set, Client *clientTable, int server_socket) {
   FD_ZERO(set);
   Client *client = clientTable;

   FD_SET(server_socket, set);
   FD_SET(STDIN_FILENO, set);
   while (client) {
      FD_SET(client->socket, set);
      client = client->next;
   }
}

void HandleData(char *buf, Client **clientTable, int client_socket, int message_len, int *frame_num) {
   int iter = 0;
   Header *header = ((Header *)buf); 
   char flag = header->flag;  
   char size = sizeof(Header);
   
   buf += sizeof(Header);
   switch(flag) {
      case INIT_CLIENT_FLAG:
         NewClient(buf, clientTable, client_socket, header); 
         break;
      case DATA_IN_FLAG:
         while (size < header->size) {
            if ((message_len =  recv(ready_socket, buf, BUF_SIZE, 0)) < 0)
            {
                perror("recv call");
            }  
            else {
                size += header->size;
            }  
         }
         if (frame_num + 1 == header->frame_num) {
            //output frame.     
         }
         else {
            //buffer frame.
         }
         break;
   }
}

void RemoveClient(Client **clientTable, int client_socket) {
   Client *prev = *clientTable;
   Client *client = *clientTable;
   int exit = 0;
   if (client->socket == client_socket) {
      *clientTable = (*clientTable)->next;
      free(client->handle);
      free(client);
   }
   else {
      while (client && !exit) {
         if (client->socket == client_socket) {
            prev->next = client->next;
            free(client->handle);
            free(client);
            exit = 1;
         }
         else {
            prev = client;
            client = client->next;
         }
      }
   }
}

void SendExit(Header *header, int client_socket) {
   int sent = 0;

   header->flag = 9;
   sent = send(client_socket, header, sizeof(Header), 0);
   if(send < 0) {
      perror("send call");
   }
   close(client_socket);
}

void NewClient(char *buf, Client **clientTable, int client_socket, Header *header) {
         char handle_buf[HANDLE_SIZE];   
         int size = *buf++;
         Client *client;

         memcpy(handle_buf, buf, size);
         handle_buf[size] = '\0';
         if (!CheckHandle(handle_buf, *clientTable)) {
            if (*clientTable) {
               client = traverse_client_table(*clientTable);
               client->next = (Client *)malloc(sizeof(Client));
               client = client->next;
            }
            else {
               *clientTable = (Client *)malloc(sizeof(Client));
               client = *clientTable;
            }
            client->next = NULL;
            client->handle = (char *)malloc(size + 1);
            client->socket = client_socket;
            memcpy(client->handle, handle_buf, size);
            client->handle[size] = '\0';
            SendNewClientResponse(client_socket, header, 2);
         }
         else {
            SendNewClientResponse(client_socket, header, 3); 
         }
}

void SendNewClientResponse(int client_socket, Header *header, char flag) {
   int sent = 0;

   header->flag = flag;
   sent = send(client_socket, header, sizeof(Header), 0);
   if(send < 0) {
      perror("send call");
   }

}

int CheckHandle(char *handle, Client *clientTable) {
   Client *temp = clientTable;
   int retVal = 0;
   while (temp && !retVal) {
      if (!strcmp(handle, temp->handle)) {
         retVal = temp->socket;
      }
      temp = temp->next;
   } 
   return retVal;
}

Client *traverse_client_table(Client *clientTable) {
   Client *temp = clientTable;
   while (temp && temp->next) {
      temp = temp->next;
   }
   return temp;
}
/* This function sets the server socket.  It lets the system
   determine the port number.  The function returns the server
   socket number and prints the port number to the screen.  */

int tcp_recv_setup(int portNumber)
{
    int server_socket= 0;
    struct sockaddr_in local;      /* socket address for local side  */
    socklen_t len= sizeof(local);  /* length of local address        */

    /* create the socket  */
    server_socket= socket(AF_INET, SOCK_STREAM, 0);
    if(server_socket < 0)
    {
      perror("socket call");
      exit(1);
    }

    local.sin_family= AF_INET;         //internet family
    local.sin_addr.s_addr= INADDR_ANY; //wild card machine address
    local.sin_port= htons(portNumber);                 //let system choose the port

    /* bind the name (address) to a port */
    if (bind(server_socket, (struct sockaddr *) &local, sizeof(local)) < 0)
      {
	perror("bind call");
	exit(-1);
      }
    
    //get the port name and print it out
    if (getsockname(server_socket, (struct sockaddr*)&local, &len) < 0)
      {
	perror("getsockname call");
	exit(-1);
      }

    printf("socket has port %d \n", ntohs(local.sin_port));
	        
    return server_socket;
}

/* This function waits for a client to ask for services.  It returns
   the socket number to service the client on.    */

int tcp_listen(int server_socket, int back_log)
{
  int client_socket= 0;
  if (listen(server_socket, back_log) < 0)
    {
      perror("listen call");
      exit(-1);
    }
  
  if ((client_socket= accept(server_socket, (struct sockaddr*)0, (socklen_t *)0)) < 0)
    {
      perror("accept call");
      exit(-1);
    }
  
  return(client_socket);
}

