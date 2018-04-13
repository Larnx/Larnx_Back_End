#define _CRT_SECURE_NO_WARNINGS
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void readRectify(VideoCapture capLeft, VideoCapture capRight, int& num_images,
	int img_width, int img_height, char* imgsLeft_directory, char* imgsRight_directory, char* extension) {

	Mat imgLeft, img_resLeft, imgRight, img_resRight;

	while ((char)waitKey(5) != 'q') { // press "q" key to escape
		waitKey(3);
		capLeft >> imgLeft;
		capRight >> imgRight;

		if (imgLeft.empty()) {
			cout << "Empty frame \n";
			break;
		}
		if (imgRight.empty()) {
			cout << "Empty frame \n";
			break;
		}

		resize(imgLeft, img_resLeft, Size(img_width, img_height));
		resize(imgRight, img_resRight, Size(img_width, img_height));

		imshow("Left Camera", imgLeft);
		imshow("Right Camera", imgRight);


		if ((char)waitKey(5) == 's') { // if press "s" key, will save screenshots
			num_images++;
			char filenameLeft[200], filenameRight[200];
			sprintf(filenameLeft, "%s\\left%d.%s", imgsLeft_directory, num_images, extension);
			sprintf(filenameRight, "%s\\right%d.%s", imgsRight_directory, num_images, extension);
			cout << "Saving img pair " << num_images << endl;
			imwrite(filenameLeft, img_resLeft);
			imwrite(filenameRight, img_resRight);
		}
	}
	capLeft.release();
	capLeft.release();
	destroyAllWindows;
}

void splitImage(Mat frame, Mat& leftFrame, Mat& rightFrame) {
	int img_size = frame.size().width;
	leftFrame = frame(cv::Range(0, frame.size().height), cv::Range(0, (img_size / 2) - 1));
	rightFrame = frame(cv::Range(0, frame.size().height), cv::Range((img_size / 2) + 1, img_size));
}

int main() {

	VideoCapture vcap(1);
	if (!vcap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}


	string left_vid, right_vid;
	char* image_folder = "cb3";
	cout << "Left video filename: ";
	cin >> left_vid;
	cout << "\n";
	cout << "Right video filename: ";
	cin >> right_vid;
	cout << "\n";

	int frame_width1 = (vcap.get(CV_CAP_PROP_FRAME_WIDTH) / 2) - 1;
	int frame_height1 = vcap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int frame_width2 = (vcap.get(CV_CAP_PROP_FRAME_WIDTH) / 2) - 1;
	int frame_height2 = vcap.get(CV_CAP_PROP_FRAME_HEIGHT);
	VideoWriter video1(left_vid, CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width1, frame_height1), true);
	VideoWriter video2(right_vid, CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width2, frame_height2), true);


	for (;;) {

		Mat frame1, leftFrame, rightFrame;
		vcap >> frame1;
		splitImage(frame1, leftFrame, rightFrame);
		imshow("Frame1", leftFrame);
		imshow("Frame2", rightFrame);
		video1.write(leftFrame);
		video2.write(rightFrame);

		char c = (char)waitKey(33);
		if (c == 27) break;
	}
	VideoCapture cap1(left_vid);
	VideoCapture cap2(right_vid);
	int num_imgs = 0;

	readRectify(cap1, cap2, num_imgs, cap1.get(CV_CAP_PROP_FRAME_WIDTH), cap1.get(CV_CAP_PROP_FRAME_HEIGHT), image_folder, image_folder, "jpg");



	return 0;

	
}
