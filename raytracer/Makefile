LIBS = -lGL -lGLU -lGLEW -lglut -L/usr/local/cuda-7.5/lib64 -lcudart -lAntTweakBar
FLAGS = -std=c++11

kernel:
	nvcc -c raytracer.cu -o kernel.o

utils:
	g++ -c utils.cpp $(FLAGS) -o utils.o

raytracer:
	make kernel
	make utils
	g++ main.cpp utils.o kernel.o $(FLAGS) $(LIBS) -o raytracer

clean:
	rm *.o raytracer
