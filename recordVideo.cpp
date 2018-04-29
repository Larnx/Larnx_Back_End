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

void splitImage(Mat frame, Mat& leftFrameFull, Mat& rightFrameFull, Mat& leftFrameCropped, Mat& rightFrameCropped) {
	int img_size = frame.size().width;
	/*leftFrame = frame(cv::Range(0, frame.size().height), cv::Range(0, (img_size / 2) - 1));
	rightFrame = frame(cv::Range(0, frame.size().height), cv::Range((img_size / 2) + 1, img_size));*/
	// Incase the camera is working correctly
	leftFrameFull = frame(cv::Range(0, frame.size().height), cv::Range(0, (img_size / 2) - 1));
	rightFrameFull = frame(cv::Range(0, frame.size().height), cv::Range((img_size / 2) + 1, img_size));
	// Incase the camera is not working
	leftFrameCropped = frame(cv::Range(0, frame.size().height - 60), cv::Range(0, (img_size / 2) - 1));
	rightFrameCropped = frame(cv::Range(60, frame.size().height), cv::Range((img_size / 2) + 1, img_size));
}

int main() {

	VideoCapture vcap(1);
	if (!vcap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}


	string left_vid, right_vid;
	//char* image_folder = "cb3";
	cout << "You do not need to include .avi" << endl;
	cout << "Left video filename: ";
	cin >> left_vid;
	cout << "\n";
	cout << "Right video filename: ";
	cin >> right_vid;
	cout << "\n";

	int frame_width1Full = (vcap.get(CV_CAP_PROP_FRAME_WIDTH) / 2) - 1;
	int frame_height1Full = vcap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int frame_width2Full = (vcap.get(CV_CAP_PROP_FRAME_WIDTH) / 2) - 1;
	int frame_height2Full = vcap.get(CV_CAP_PROP_FRAME_HEIGHT);

	int frame_width1Cropped = (vcap.get(CV_CAP_PROP_FRAME_WIDTH) / 2) - 1;
	int frame_height1Cropped = (vcap.get(CV_CAP_PROP_FRAME_HEIGHT) - 60);
	int frame_width2Cropped = (vcap.get(CV_CAP_PROP_FRAME_WIDTH) / 2) - 1;
	int frame_height2Cropped = (vcap.get(CV_CAP_PROP_FRAME_HEIGHT) - 60);

	VideoWriter video1Full(left_vid + "left_full.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width1Full, frame_height1Full), true);
	VideoWriter video2Full(right_vid + "right_full.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width2Full, frame_height2Full), true);

	VideoWriter video1Cropped(left_vid + "left_cropped.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width1Cropped, frame_height1Cropped), true);
	VideoWriter video2Cropped(right_vid + "right_cropped.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width2Cropped, frame_height2Cropped), true);


	for (;;) {

		Mat frame1, leftFrameF, rightFrameF, leftFrameC, rightFrameC;
		vcap >> frame1;
		imshow("base frame", frame1);
		splitImage(frame1, leftFrameF, rightFrameF, leftFrameC, rightFrameC);
		imshow("FrameLF", leftFrameF);
		imshow("FrameRF", rightFrameF);
		imshow("FrameLC", leftFrameC);
		imshow("FrameRC", rightFrameC);

		video1Full.write(leftFrameF);
		video2Full.write(rightFrameF);

		video1Cropped.write(leftFrameC);
		video2Cropped.write(rightFrameC);

		char c = (char)waitKey(33);
		if (c == 27) break;
	}

	/*left_vid = left_vid + ".avi";
	right_vid = right_vid + ".avi";
	VideoCapture cap1(left_vid);
	VideoCapture cap2(right_vid);

	left_vid = "Cropped" + left_vid;
	right_vid = "Cropped" + right_vid;
	VideoCapture cap3(left_vid);
	VideoCapture cap4(right_vid);*/

	//int num_imgs = 0;

	//readRectify(cap1, cap2, num_imgs, cap1.get(CV_CAP_PROP_FRAME_WIDTH), cap1.get(CV_CAP_PROP_FRAME_HEIGHT), image_folder, image_folder, "jpg");



	return 0;


}