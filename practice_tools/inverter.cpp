#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>

#define GRAY	0
#define INV		1
#define THRESH	2
#define EDGE	3

#define BYTE 	8

#define MAX_RGB		765
#define MAX_GRAY	255

using namespace cv;

typedef struct {
	uchar 	blue;
	uchar 	green;
	uchar 	red;
} pixel;

// ----- FUNCTION HEADERS ----- //
uint pixelValue(pixel p);
void getSurroundingRGB(pixel ret[8], Mat img, ulong curPixel);
uint edgeValue(pixel* pixels);
// ---------------------------- //

int main (int argc, char** argv) {
	Mat 	image;
	Mat 	output[3];
	uchar 	highest = 0;
	bool 	flags[] = {false, false, false, false};
	uint 	maxThresh = MAX_RGB, threshold, edge, pixelSize = 3;
	size_t 	option;
	pixel	p = {0, 0, 0};
	pixel	edgeArray[8];
	
    for (option = 1; option < argc && argv[option][0] == '-'; option++) {
        switch (argv[option][1]) {
		case 'g':
			flags[GRAY] = true;
			pixelSize = 1;
			maxThresh = MAX_GRAY;
			break;
        case 'i': 
			flags[INV] = true;
			break;
        case 't': 
			flags[THRESH] = true;
			if (sscanf(argv[++option], "%d", &threshold) < 1) {
				fprintf(stderr, "Invalid argument for -t\n");
				return -1;
			}
			fprintf(stderr, "Value received for -t: %d\n", threshold);
			break;
        case 'e': 
			flags[EDGE] = true;
			if (sscanf(argv[++option], "%d", &edge) < 1) {
				fprintf(stderr, "Invalid argument for -e\n");
				return -1;
			}
			break;
        default:
            fprintf(stderr, "Incorrect usage\n");
            exit(EXIT_FAILURE);
        }   
    }

	if (threshold > maxThresh) {
		threshold = maxThresh;
	}
	if (edge > maxThresh) {
		edge = maxThresh;
	}
	
	if (flags[GRAY]) {
		image = imread("./mona-lisa.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	} else {
		image = imread("./mona-lisa.jpg", CV_LOAD_IMAGE_COLOR);	
	}

	if (!image.data) {
		fprintf(stderr, "No image data\n");
		return -1;
	}

	if (flags[GRAY]) {
		output[0].create(Size(image.cols, image.rows), CV_8UC1);
		output[1].create(Size(image.cols, image.rows), CV_8UC1);
		output[2].create(Size(image.cols, image.rows), CV_8UC1);
	} else {
		output[0].create(Size(image.cols, image.rows), CV_8UC3);
		output[1].create(Size(image.cols, image.rows), CV_8UC3);
		output[2].create(Size(image.cols, image.rows), CV_8UC3);
	}

	printf("------ Input Image Data ------\n");
	printf("Rows: %d Columns: %d\n", image.rows, image.cols);
	printf("Number of dimensions: %d\n", image.dims);
	printf("First byte of image data: %d\n", image.data[0]);

	// Inverted image
	if (flags[INV]) {
		for (int i = 0; i < image.rows*image.cols*pixelSize; i++) {
			if (image.data[i] > highest) highest = image.data[i];
			output[0].data[i] = 255 - image.data[i];
		}
	
		printf("Highest data value: %d\n", highest);
	
		printf("------ Inverted Image Data ------\n");
		printf("Rows: %d Columns: %d\n", output[0].rows, output[0].cols);
		printf("Number of dimensions: %d\n", output[0].dims);
		printf("First byte of output data: %d\n", output[0].data[0]);
	
		namedWindow("Inverted Image", WINDOW_NORMAL);
		imshow("Inverted Image", output[0]);
	}
	
	// Thresholded image
	if (flags[THRESH]) {
		for (int i = 0; i < image.rows*image.cols*pixelSize; i += pixelSize) {
			// Reset pixel data
			p = {0, 0, 0};
			// Copy image.data (1 or 3 bytes) into a pixel
			memcpy(&p, (image.data + i), pixelSize);
			if (pixelValue(p) >= threshold) {
				// Higher than threshold -> white
				p.blue = 255;
				p.green = 255;
				p.red = 255;
			} else {
				// Lower than threshold -> black
				p.blue = 0;
				p.green = 0;
				p.red = 0;
			}
			// Put the new pixel (black or white) into output.data
			memcpy((output[1].data + i), &p, pixelSize);
		}
		
		namedWindow("Thresholded Image", WINDOW_NORMAL);
		imshow("Thresholded Image", output[1]);
	}
	
	// Edge-detected image
	if (flags[EDGE]) {
		if (flags[GRAY]) {
			fprintf(stderr, "Please only use edge-detection with RGB images\n");
			return -1;
		}
		for (int i = 0; i < image.rows*image.cols*pixelSize; i += pixelSize) {
			// Reset pixel data
			p = {0, 0, 0}; 
			// Find the pixels around the current one
			getSurroundingRGB(edgeArray, image, i);
			if (edgeValue(edgeArray) > edge) {
				// Higher than threshold -> white
				p.blue = 255;
				p.green = 255;
				p.red = 255;
			} else {
				// Lower than threshold -> black
				p.blue = 0;
				p.green = 0;
				p.red = 0;
			}
			// Put the new pixel (black or white) into output.data
			memcpy((output[2].data + i), &p, pixelSize);
		}
		
		namedWindow("Edge-detected Image", WINDOW_NORMAL);
		imshow("Edge-detected Image", output[2]);
	}
	
	namedWindow("Original Image", WINDOW_NORMAL);
	imshow("Original Image", image);
	
	waitKey(0);
 
	return 0;
} 

uint pixelValue(pixel p) {	
	uint val;
	
	val = p.blue + p.green + p.red;
	
	return val;
}

// Returns an array of the RGB pixels surround the one at the given point in the given Mat img
void getSurroundingRGB(pixel ret[8], Mat img, ulong curPixel) {
	Vec3b pixels[8];
	uint curRow, curCol, i;
	
	// Row = pixel# / # of cols, rounded down
	curRow = curPixel / img.cols;
	
	// Col = pixel# - (row# * # of cols)
	curCol = curPixel - (curRow * img.cols);
	
	if (curRow == 0 || curCol == 0 || curRow >= (img.rows - 1) || curCol >= (img.cols - 1)) {
		for (i = 0; i < 8; i++) {
			ret[i].blue = 0;
			ret[i].green = 0;
			ret[i].red = 0;
		}
	} else {
		pixels[0] = img.at<Vec3b>(curRow - 1, curCol - 1);
		pixels[1] = img.at<Vec3b>(curRow - 1, curCol);
		pixels[2] = img.at<Vec3b>(curRow - 1, curCol + 1);
		pixels[3] = img.at<Vec3b>(curRow, curCol + 1);
		pixels[4] = img.at<Vec3b>(curRow + 1, curCol + 1);
		pixels[5] = img.at<Vec3b>(curRow + 1, curCol);
		pixels[6] = img.at<Vec3b>(curRow + 1, curCol - 1);
		pixels[7] = img.at<Vec3b>(curRow, curCol - 1);
		
		for (i = 0; i < 8; i++) {
			ret[i].blue = pixels[i].val[0];
			ret[i].green = pixels[i].val[1];
			ret[i].red = pixels[i].val[2];
			//fprintf(stderr, "--- OBTAINED PIXEL VALUE %3d %3d %3d ---\n", ret[i].blue, ret[i].green, ret[1].red);
		}
	}
	
	return;
}

uint edgeValue(pixel* pixels) {
	uint ret, high = 0, low = 765, tmp, i;
	
	// Get min/max pixel values
	for (i = 0; i < 8; i++) {
		tmp = pixelValue(pixels[i]);
		if (tmp > high) high = tmp;
		if (tmp < low) low = tmp;
	}
	
	// Calculate edge value: high - low
	ret = high - low;
	//fprintf(stderr, "--- OBTAINED EDGE VALUE %3d ---\n", ret);
	
	return ret;
}