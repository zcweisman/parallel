#ifndef CONSTANTS_CAPNCURSES
#define CONSTANTS_CAPNCURSES

// ----- Macros ----- //
#define COL(x) (COL0 + (x)*COL_WID)
#define ROW(x) (ROW0 + (x)*ROW_HGT)

#define WHITE     (COLOR_PAIR(0))
#define RED       (COLOR_PAIR(1))
#define GREEN     (COLOR_PAIR(2))
#define YELLOW    (COLOR_PAIR(3))
#define CRITICAL  (COLOR_PAIR(4))

// ----- Data Structures ----- //
struct nodeInfo {
   int temp;
   double cpu;
   int ram;
};

// ----- Actual constants ----- //
#define NUM_NODES       79

#define LINELENGTH      120

#define NODE            "Node #"
#define CORETEMP        "Temperature (C)"
#define CPU             "CPU Usage (\%)"
#define GPU             "GPU Usage (\%)"
#define RAM             "RAM Usage (cur/max)"

#define TEMP_AVG        "Average core temp:"
#define CPU_AVG         "Average CPU usage:"
#define GPU_AVG         "Average GPU usage:"
#define RAM_AVG         "Average RAM usage:"

#define COL0      5
#define COL_WID   22
#define COL_PAD   2

#define ROW0      5
#define ROW_HGT   2

#define AGGREGATE_ROWS  3

// ----- Files where data is located ----- //
// Temperature
#define FTEMP_CRIT   "/sys/class/hwmon/hwmon0/temp1_crit"
#define FTEMP_CUR    "/sys/class/hwmon/hwmon0/temp1_input"

// CPU
#define FCPU      "/proc/stat"
#define CPUTOT    "cpu"
#define CPU_IDLE  4

// RAM usage
#define FRAM      "/proc/meminfo"
#define RAMTOT    "MemTotal:"
#define RAMFREE   "MemFree:"

#endif
