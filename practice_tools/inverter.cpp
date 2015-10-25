#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main (int argc, char** argv) {
	Mat image;
	Mat output;
	uchar highest = 0;
	image = imread("./mona-lisa.jpg", CV_LOAD_IMAGE_COLOR);

	if (!image.data) {
		printf("No image data\n");
		return -1;
	}

	output.create(Size(image.cols, image.rows), CV_8UC3);

	printf("------ Input Image Data ------\n");
	printf("Rows: %d Columns: %d\n", image.rows, image.cols);
	printf("Number of dimensions: %d\n", image.dims);
	printf("First byte of image data: %d\n", image.data[0]);

	for (int i = 0; i < image.rows*image.cols*3; i++) {
		if (image.data[i] > highest) highest = image.data[i];
		output.data[i] = 255 - image.data[i];
	}

	printf("Highest data value: %d\n", highest);

	printf("------ Output Image Data ------\n");
	printf("Rows: %d Columns: %d\n", output.rows, output.cols);
	printf("Number of dimensions: %d\n", output.dims);
	printf("First byte of output data: %d\n", output.data[0]);

	namedWindow("Inverted Image", WINDOW_NORMAL);
	imshow("Inverted Image", output);
	namedWindow("Original Image", WINDOW_NORMAL);
	imshow("Original Image", image);
	
	waitKey(0);
 
	return 0;
} 
