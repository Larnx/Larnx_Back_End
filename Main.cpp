// Main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <cstring>
#include <stdlib.h>
#include <map>

using namespace std;
using namespace cv;

double const	fps = 30;
double const	timeStep = 1 / fps;


void saveFrame(string Video_Path, string Output_Directory_Path, double Selected_Frame_TimeStamp)
{
	Mat frame;
	VideoCapture cap(Video_Path);

	namedWindow("Video Capture", WINDOW_NORMAL);

	int frameCount = 0;
	double currentTime;
	while ((char)waitKey(1) != 'q')
	{
		frameCount++; 
		currentTime = timeStep*frameCount;
		double currenttime_ms = 1000 * currentTime;

		cap >> frame;
		if (frame.empty()) break;

		if (round(currenttime_ms) == Selected_Frame_TimeStamp) {		// save the frame if its timestamp matches the desired timestamp. 
			imshow("video capture", frame);							    // show us the frame to be sure ;) 
			imwrite(Output_Directory_Path + "n.jpg", frame);			// and save it 
			break;														// no need to keep going. we can modify this to save a arbitrary number of frames
		}
	}
}

void ThresholdHSV(string Video_Path, string Output_Directory_Path) {

	// get video
	Mat bright, brightHSV;
	// VideoCapture cap(Video_Path);
	VideoCapture cap(Video_Path);
	/*End Video Parameters*/

	namedWindow("Video Capture", WINDOW_NORMAL);
	namedWindow("Object Detection", WINDOW_NORMAL);

	/* Current Function: Segments green from video, returns pixel data vs time stamps for UI to plot to console, saves segmented vdo */
	/* Future Functions: Track green residue with bound boxes, returns ratio pixel data vs time stamp*/
	//create an ofstream for the file output 
	ofstream outputFile;
	// create a name for the file output
	string filename = "GreenPixels.csv";
	// create the .csv file
	outputFile.open(Output_Directory_Path + filename);
	// write the file headers for csv
	outputFile << "Frame Count" << "," << "Time (sec)" << "," << "Pixel Intensity" << endl;

	// Write new video
	string filenameVDO = "New.avi";

	// get frame size
	Size frameSize(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));  // frame width = 1280, height = 720
	int fps = cap.get(CAP_PROP_FPS); // 30 frames per second

	// create output window
	VideoWriter writer;
	writer.open(Output_Directory_Path + filenameVDO, 0, fps, frameSize, 1);

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
		//cout << cap.get(CAP_PROP_POS_MSEC) / 1000 << endl << cv::sum(resultHSV)[0] << endl;


		//cout << "\{ \"x:\"\"" << cap.get(CAP_PROP_POS_MSEC) / 1000 << "\",\"y:\"" << cv::sum(resultHSV)[0] << "\"\}" << endl;
		// JSON Format
		//'{ "name":"John", "age":30, "city":"New York"}'


		cout << cv::sum(resultHSV)[0] << endl;

		writer.write(resultHSV);

		imshow("Video Capture", bright);
		imshow("Object Detection", resultHSV);

	}
	// close the output file
	outputFile.close();
}


int main(int argc, char *argv[]) {

	// First argument determines behavior
	int method = atoi(argv[1]);

	// Initialize Variables - Not all will be used 
	string Video_Path;
	string Output_Directory_Path;
	double Selected_Frame_TimeStamp;
	double Trim_Start, Trim_End;

	// Keep a dictionary of methods, for error throwing and reference. 
	std::map<int, string> first;
	first[1] = "Process Video";
	first[2] = "Save Frame";
	first[3] = "Trim Video";


	switch (method) 
	{
		case 1 :	// 1: Process Video
		{
			if (argc != 4) {
				printf("Invalid usage: Method %s in process %s", first[1], argv[0]);
			}
			else {
				Video_Path = argv[2];
				Output_Directory_Path = argv[3];
			}

			ThresholdHSV(Video_Path,Output_Directory_Path);

			break;
		}
		case 2 :	// 2: Save Selected Frame
		{
			if (argc != 5) {
				printf("Invalid usage: Method %s in process %s", first[2], argv[0]);
			}
			else {
				Video_Path = argv[2];
				Output_Directory_Path = argv[3];
				Selected_Frame_TimeStamp = atof(argv[4]);
			}

			saveFrame(Video_Path,Output_Directory_Path,Selected_Frame_TimeStamp);

			break;
		}
		case 3 :	// 3: Trim Video
		{
			if (argc != 5) {
				printf("Invalid usage: Method %s in process %s", first[3], argv[0]);
			}
			else {
				Video_Path = argv[2];
				Output_Directory_Path = argv[3];
				Trim_Start = atof(argv[4]);
				Trim_End = atof(argv[5]);
			}

			// Add function call here 

			break;
		}
		default: 
			break;
	}

	return 0;
}
