#include <stdlib.h>
#include <stdio.h>
#include <curses.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include "constants.h"

static void finish (int sig);

int main(int argc, char *argv[]) {
   FILE *file, *logFile;
   char fileContent[LINELENGTH];
   char field[20];
   struct nodeInfo nodes[NUM_NODES];
   struct nodeInfo average = {0, 0.0, 0};
   int tempCrit, ramMax;
   int i;

   struct timespec req = {0, 500000000};
   struct timespec rem = {0, 0};

   (void) signal(SIGINT, finish);      /* arrange interrupts to terminate */

   (void) initscr();      /* initialize the curses library */

   if (has_colors())
   {
      start_color();

      // Color assignments
      init_pair(1, COLOR_RED,       COLOR_BLACK);
      init_pair(2, COLOR_GREEN,     COLOR_BLACK);
      init_pair(3, COLOR_YELLOW,    COLOR_BLACK);
      init_pair(4, COLOR_BLACK,     COLOR_RED);
   }

   // Define table framework (headers, node# column)
   // node#
   mvaddstr(ROW(0), COL(0) + COL_PAD, NODE);
   // Temperature
   mvaddstr(ROW(0), COL(1) + COL_PAD, CORETEMP);
   // cpu
   mvaddstr(ROW(0), COL(2) + COL_PAD, CPU);
   // ram
   mvaddstr(ROW(0), COL(3) + COL_PAD, RAM);

   // node# column
   mvaddstr(ROW(1), COL(0) + COL_PAD, TEMP_AVG);
   mvaddstr(ROW(2), COL(0) + COL_PAD, CPU_AVG);
   mvaddstr(ROW(3), COL(0) + COL_PAD, RAM_AVG);
   for (i = 1; i <= NUM_NODES; i++) {
      sprintf(field, "%d", i);
      mvaddstr(ROW(i + AGGREGATE_ROWS), COL(0) + COL_PAD, field);
   }

   refresh();

   // HOST: get total available ram, critical temperature, etc
   // Critical temp
   file = fopen(FTEMP_CRIT, "r");
   fgets(fileContent, LINELENGTH, file);
   sscanf(fileContent, "%d", &tempCrit);
   // Remove "000" from the end to get temp in C
   tempCrit /= 1000;
   fclose(file);

   // Max available RAM
   file = fopen(FRAM, "r");
   fgets(fileContent, LINELENGTH, file);
   // Go through lines until we get the right one
   while(!strstr(fileContent, RAMTOT)) {
      fgets(fileContent, LINELENGTH, file);
   }
   sscanf(fileContent, "MemTotal: %d kB", &ramMax);
   fclose(file);

   // Update table fields
   while (TRUE) {
      // TODO: Get node data via network

      // Print mechanical data
      // Temperature
      for (i = 0; i < NUM_NODES; i++) {
         sprintf(field, "%5d", nodes[i].temp);
         if (nodes[i].temp > 0.65*((double)tempCrit)) attrset(YELLOW);
         if (nodes[i].temp > 0.80*((double)tempCrit)) attrset(RED);
         if (nodes[i].temp > 0.95*((double)tempCrit)) attrset(CRITICAL);
         mvaddstr(ROW(i + 1 + AGGREGATE_ROWS), COL(1) + COL_PAD, field);
         attrset(WHITE);
      }
      sprintf(field, "%5d", average.temp);
      if (average.temp > 0.65*((double)tempCrit)) attrset(YELLOW);
      if (average.temp > 0.80*((double)tempCrit)) attrset(RED);
      if (average.temp > 0.95*((double)tempCrit)) attrset(CRITICAL);
      mvaddstr(ROW(1), COL(1) + COL_PAD, field);
      attrset(WHITE);

      // CPU
      for (i = 0; i < NUM_NODES; i++) {
         sprintf(field, "%5f", nodes[i].cpu);
         if (nodes[i].cpu > 35.0) attrset(YELLOW);
         if (nodes[i].cpu > 75.0) attrset(RED);
         mvaddstr(ROW(i + 1 + AGGREGATE_ROWS), COL(2) + COL_PAD, field);
         attrset(WHITE);
      }
      sprintf(field, "%5f", average.cpu);
      if (average.cpu > 35) attrset(YELLOW);
      if (average.cpu > 75) attrset(RED);
      mvaddstr(ROW(2), COL(2) + COL_PAD, field);
      attrset(WHITE);

      // RAM
      for (i = 0; i < NUM_NODES; i++) {
         sprintf(field, "%5d", nodes[i].ram);
         if (nodes[i].ram > 0.5*((double)ramMax)) attrset(YELLOW);
         if (nodes[i].ram > 0.75*((double)ramMax)) attrset(RED);
         mvaddstr(ROW(i + 1 + AGGREGATE_ROWS), COL(3) + COL_PAD, field);
         attrset(WHITE);
      }
      sprintf(field, "%5d", average.ram);
      if (average.ram > 0.5*((double)ramMax)) attrset(YELLOW);
      if (average.ram > 0.75*((double)ramMax)) attrset(RED);
      mvaddstr(ROW(3), COL(3) + COL_PAD, field);
      attrset(WHITE);

      refresh();

      // wait a sec
      //sleep(1);
      nanosleep(&req, &rem);
   }

   finish(0);               /* we're done */
}

static void finish (int sig) {
   endwin();

   /* do your non-curses wrapup here */

   exit(0);
}
