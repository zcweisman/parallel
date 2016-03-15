MFLAGS = -std=c++11
CFLAGS = -std=c++11
MCC = mpic++
CC = g++

CLIBS = -lGL -lGLU -lGLEW -lglut -I/usr/local/cuda-6.5/include -L/usr/local/cuda-6.5/lib -lcudart -L/home/ubuntu/Cluster/AntTweakBar/lib -lAntTweakBar -I/home/ubuntu/Cluster/AntTweakBar/include

kernel:
	nvcc -c raytracer.cu -o ./lib/kernel.o

utils:
	$(CC) -c utils.cpp $(CFLAGS) -I/usr/local/cuda-6.5/include -o ./lib/utils.o /usr/local/cuda-6.5/lib/libcudart_static.a

application:
	$(MCC) application.cpp ./lib/utils.o ./lib/kernel.o $(CFLAGS) $(CLIBS) -o ./bin/application

clean:
	rm ./bin/application
