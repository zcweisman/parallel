CC=g++
CFLAGS= -ansi -pedantic -Wno-deprecated -g -std=c++11
LIBS=-I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching


linux:
	$(CC) $(CFLAGS) ./practice_tools/inverter.cpp $(LIBS)

camera:
	$(CC) $(CFLAGS) ./practice_tools/camerafeed.cpp $(LIBS)
