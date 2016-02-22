#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include "constants.h"

#define TRUE   1
#define FALSE  0

int main(int argc, char *argv[]) {
   FILE *temperature, *cpu, *ram;
   char fileContent[LINELENGTH];
   struct nodeInfo node = {0, 0.0, 0};
   int maxRam;
   int user, nice, kernel, idle, io, irq, sirq, total, used;
   int i;

   struct timespec req = {0, 500000000};
   struct timespec rem = {0, 0};

   // Get max ram
   ram = fopen(FRAM, "r");
   fgets(fileContent, LINELENGTH, ram);
   // Go through lines until we get the right one
   while(!strstr(fileContent, RAMTOT)) {
      fgets(fileContent, LINELENGTH, ram);
   }
   sscanf(fileContent, "MemTotal: %d kB", &maxRam);
   fclose(ram);

   // Get data forever
   while (TRUE) {
      // NODE: get current used ram, current temp, current CPU
      // Current temp
      temperature = fopen(FTEMP_CUR, "r");
      fgets(fileContent, LINELENGTH, temperature);
      sscanf(fileContent, "%d", &node.temp);
      // Remove "000" from the end to get temp in C
      node.temp /= 1000;
      fclose(temperature);

      // Current CPU usage
      cpu = fopen(FCPU, "r");
      fgets(fileContent, LINELENGTH, cpu);
      // Go through lines until we get the right one
      while(!strstr(fileContent, CPUTOT)) {
         fgets(fileContent, LINELENGTH, cpu);
      }
      sscanf(fileContent, "cpu %d %d %d %d %d %d %d", &user, &nice, &kernel, &idle, &io, &irq, &sirq);
      total = user + nice + kernel + idle + io + irq + sirq;
      used = total - idle;
      node.cpu = ((double)used / (double)total) * 100.0;
      fclose(cpu);

      // Current used RAM
      ram = fopen(FRAM, "r");
      fgets(fileContent, LINELENGTH, ram);
      // Go through lines until we get the right one
      while(!strstr(fileContent, RAMFREE)) {
         fgets(fileContent, LINELENGTH, ram);
      }
      sscanf(fileContent, "MemFree: %d kB", &node.ram);
      node.ram = maxRam - node.ram;
      fclose(ram);

      // Print data for now
      printf("temp = %d C   cpu usage = %4f   ram used = %d kB\n", node.temp, node.cpu, node.ram);

      // TODO: Send node data via network



      // Wait a sec
      //sleep(1);
      nanosleep(&req, &rem);
   }

   return;
}
