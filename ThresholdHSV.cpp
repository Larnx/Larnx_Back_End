// ThreholdHSV.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <cstring>
#include <stdlib.h>

///////////////////////////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

	//Vec3b bgrPixel(40, 158, 16);
	// Create Mat object from vector since cvtColor accepts a Mat object
	//Mat3b bgr(bgrPixel);

	//Convert pixel values to other color spaces.
	//Mat3b hsv, ycb, lab;
	// cvtColor(bgr, ycb, COLOR_BGR2YCrCb);
	//cvtColor(bgr, hsv, COLOR_BGR2HSV);
	// cvtColor(bgr, lab, COLOR_BGR2Lab);
	//Get back the vector from Mat
	//Vec3b hsvPixel(hsv.at<Vec3b>(0, 0));
	//Vec3b ycbPixel(ycb.at<Vec3b>(0, 0));
	//Vec3b labPixel(lab.at<Vec3b>(0, 0));

	//int thresh = 70;

	vector<double> timeStamp;
	// create an ofstream for the file output (see the link on streams for
	// more info)
	ofstream outputFile;
	// create a name for the file output
	string filename = "GreenPixels.csv";
	// create the .csv file
	outputFile.open(filename);

	// write the file headers
	outputFile << "Frame Count" << "," << "Time (sec)" << "," << "Pixel Intensity" << endl;

	// Write new video
	string filenameVDO = "New Clip 64 AS.avi";

	Mat bright, brightHSV;
	VideoCapture cap("Clip 64 AS.mp4");

	// get frame size
	Size frameSize(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	int fps = 30; // 30 frames per second

	
	namedWindow("Video Capture", WINDOW_NORMAL);
	namedWindow("Object Detection", WINDOW_NORMAL);
	
	

	/*cv::Scalar minBGR = cv::Scalar(bgrPixel.val[0] - thresh, bgrPixel.val[1] - thresh, bgrPixel.val[2] - thresh);
	cv::Scalar maxBGR = cv::Scalar(bgrPixel.val[0] + thresh, bgrPixel.val[1] + thresh, bgrPixel.val[2] + thresh);

	cv::Mat maskBGR, resultBGR;
	cv::inRange(bright, minBGR, maxBGR, maskBGR);
	cv::bitwise_and(bright, bright, resultBGR, maskBGR);*/


	/*cv::Scalar minYCB = cv::Scalar(ycbPixel.val[0] - thresh, ycbPixel.val[1] - thresh, ycbPixel.val[2] - thresh)
	cv::Scalar maxYCB = cv::Scalar(ycbPixel.val[0] + thresh, ycbPixel.val[1] + thresh, ycbPixel.val[2] + thresh)

	cv::Mat maskYCB, resultYCB;
	cv::inRange(brightYCB, minYCB, maxYCB, maskYCB);
	cv::bitwise_and(brightYCB, brightYCB, resultYCB, maskYCB);

	cv::Scalar minLAB = cv::Scalar(labPixel.val[0] - thresh, labPixel.val[1] - thresh, labPixel.val[2] - thresh)
	cv::Scalar maxLAB = cv::Scalar(labPixel.val[0] + thresh, labPixel.val[1] + thresh, labPixel.val[2] + thresh)

	cv::Mat maskLAB, resultLAB;
	cv::inRange(brightLAB, minLAB, maxLAB, maskLAB);
	cv::bitwise_and(brightLAB, brightLAB, resultLAB, maskLAB);*/

	//cv2::imshow("Result BGR", resultBGR)
	//cv2::imshow("Result HSV", resultHSV);
	//cv2::imshow("Result YCB", resultYCB)
	//cv2::imshow("Output LAB", resultLAB)
	VideoWriter writer;
	writer.open(filenameVDO, 0, fps, frameSize, 1);

	while ((char)waitKey(1) != 'q') {
		cap >> bright;
		if (bright.empty())
			break;


		// convert to HSV color space
		cvtColor(bright, brightHSV, COLOR_BGR2HSV);

		//-- Detect the object based on HSV Range Values
		//Scalar minHSV = Scalar(hsvPixel.val[0] - 40, hsvPixel.val[1] - 40, hsvPixel.val[2] - 40);
		//Scalar maxHSV = Scalar(hsvPixel.val[0] + 40, hsvPixel.val[1] + 40, hsvPixel.val[2] + 40);

		Scalar minHSV = Scalar(65, 17, 140);
		Scalar maxHSV = Scalar(123, 269, 198);

		Mat maskHSV, resultHSV;
		inRange(brightHSV, minHSV, maxHSV, maskHSV);
		bitwise_and(brightHSV, brightHSV, resultHSV, maskHSV);

		// get time stamps and sum pixels
		outputFile << cap.get(CAP_PROP_POS_FRAMES) << "," << cap.get(CAP_PROP_POS_MSEC)/1000 << "," << cv::sum(resultHSV)[0] << endl;
		
		writer.write(resultHSV);

		imshow("Video Capture", bright);
		imshow("Object Detection", resultHSV);
		
	}
	// close the output file
	outputFile.close();

	
	return 0;
}
