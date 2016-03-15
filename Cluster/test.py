import re
import subprocess
import os
import time
import socket

MEM_FILE_NAME = "/proc/meminfo"
TEMP_FILE_NAME = "/sys/class/hwmon/hwmon0/temp1_input"
sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
port = 5001
if host == "master":
    sk.bind((host, port))
    sk.listen(5)
    while(1):
	print 15 * "-" + "MASTER" + 15 * "-"
	mem_file = open(MEM_FILE_NAME)
	temp_file = open(TEMP_FILE_NAME)
	line1 = mem_file.readline()
	line2 = mem_file.readline()
	max_usage = 0
	total_mem = re.search("[0-9]+\s[a-z][B]", line1)
	free_mem = re.search("[0-9]+\s[a-z][B]", line2)
	print "Total Memory: " + total_mem.group(0)
	print "Free Memory: " + free_mem.group(0)
	output = subprocess.check_output("ps -eo pcpu,args", shell=True)
	output = output.split('\n')
	usage = "Process not Running!"
	for s in output:
	    if re.search("(\/home\/ubuntu\/Cluster\/bin\/application)", s):
		if len(s.split("mpirun")) <= 1:
		    usage = re.findall("\d+\.\d+", s)[0]
	print "CPU Usage: " + str(usage)
	temp_line = temp_file.readline()
	temp = re.match("[0-9]{2}", temp_line)
	print "CPU Temperature: " + str(temp.group(0))
        if int(temp.group(0)) > max_usage:
            max_usage = int(temp.group(0))
	print "Max CPU Usage: " + str(max_usage)
	mem_file.close()
	temp_file.close()
	c, addr = sk.accept()
	res = sk.recv(1024)
        print "Hello!"
        while res:
	    res = res.split(",")		
	    print "Total Memory: " + res[0]
	    print "Free Memory: " + res[1]
	os.system("clear");
else:
	sk.connect(("master", port))
	while(1):
		mem_file = open(MEM_FILE_NAME)
		temp_file = open(TEMP_FILE_NAME)
		line1 = mem_file.readline()
		line2 = mem_file.readline()
		max_usage = 0
		total_mem = re.search("[0-9]+\s[a-z][B]", line1)
		free_mem = re.search("[0-9]+\s[a-z][B]", line2)
		output = subprocess.check_output("ps -eo pcpu,args", shell=True)
		output = output.split('\n')
		usage = "Process not Running!"
		for s in output:
		    if re.search("(\/home\/ubuntu\/Cluster\/bin\/application)", s):
			if len(s.split("mpirun")) <= 1:
			    usage = re.findall("\d+\.\d+", s)[0]
		temp_line = temp_file.readline()
		temp = re.match("[0-9]{2}", temp_line)
		if int(temp.group(0)) > max_usage:
		    max_usage = int(temp.group(0))
	output = str(total_mem.group(0)) + "," + str(free_mem.group(0)) + "," + usage + "," + str(max_usage) + "," + str(temp.group(0))
	sk.send(output)
	mem_file.close()
	temp_file.close()
	time.sleep(1)
