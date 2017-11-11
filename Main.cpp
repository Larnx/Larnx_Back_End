// Main.cpp

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

void saveFrame(Mat frame, VideoCapture cap, double Selected_Frame_TimeStamp, string Output_Path) {

	while ((char)waitKey(1) != 'q')
	{
		double ms = 1000 * cap.get(CAP_PROP_POS_MSEC); // now in seconds
		double T = cap.get(CAP_PROP_FPS);
		double F = cap.get(CAP_PROP_FRAME_COUNT);

		cap >> frame;
		if (frame.empty()) break;


		if (round(ms) == Selected_Frame_TimeStamp) {		// Save the frame if its timestamp matches the desired timestamp. 
															//imshow("Video Capture", frame);							// Show us the frame to be sure ;) 
			imwrite(Output_Path, frame);								// And save it 
			break;														// No need to keep going. We can modify this to save a arbitrary number of frames
		}

	}
}

void ThresholdHSV(VideoCapture cap, Mat bright, Mat brightHSV) {
	/* Current Function: Segments green from video, returns pixel data vs time stamps for UI to plot to console, saves segmented vdo */
	/* Future Functions: Track green residue with bound boxes, returns ratio pixel data vs time stamp*/
	//create an ofstream for the file output 
	ofstream outputFile;
	// create a name for the file output
	string filename = "GreenPixels.csv";
	// create the .csv file
	outputFile.open(filename);
	// write the file headers for csv
	outputFile << "Frame Count" << "," << "Time (sec)" << "," << "Pixel Intensity" << endl;

	// Write new video
	string filenameVDO = "New Clip 64 AS.avi";

	// get frame size
	Size frameSize(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));  // frame width = 1280, height = 720
	int fps = cap.get(CAP_PROP_FPS); // 30 frames per second

	// create output window
	VideoWriter writer;
	writer.open(filenameVDO, 0, fps, frameSize, 1);

	while ((char)waitKey(1) != 'q') {
		cap >> bright; // frame width = 1280, height = 720
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
		outputFile << cap.get(CAP_PROP_POS_FRAMES) << "," << cap.get(CAP_PROP_POS_MSEC) / 1000 << "," << cv::sum(resultHSV)[0] << endl;
		// print to console
		cout << cap.get(CAP_PROP_POS_MSEC) / 1000 << " , " << cv::sum(resultHSV)[0] << endl;

		writer.write(resultHSV);

		imshow("Video Capture", bright);
		imshow("Object Detection", resultHSV);

	}
	// close the output file
	outputFile.close();

}

int main(int argc, char *argv[]) {

	/* Video Parameters*/
	string Video_Path;					// Electron tells us where the video is located...
	string Output_Path;					// ... And Electron tells us where to write the output
	double Selected_Frame_TimeStamp;	// Electron specifies which frame it wants as a timestamp in milliseconds

	if (argc != 4) {
		printf("Invalid usage: %s filename", argv[0]);
	}
	else {
		Video_Path = argv[1];
		Output_Path = argv[2];
		Selected_Frame_TimeStamp = atof(argv[3]);
	}

	// get video
	Mat bright, brightHSV;
	// VideoCapture cap(Video_Path);
	VideoCapture cap("Clip 64 AS.mp4");
	/*End Video Parameters*/

	namedWindow("Video Capture", WINDOW_NORMAL);
	namedWindow("Object Detection", WINDOW_NORMAL);
	

	// UI calls save frame
	saveFrame(bright, cap, Selected_Frame_TimeStamp, Output_Path);

	// UI calls ThresholdHSV
	ThresholdHSV(cap, bright, brightHSV);

	return 0;
}
