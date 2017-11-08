#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdlib.h>
using namespace std;
using namespace cv;

double const		 fps = 30;
double const	timeStep = 1 / fps; 

int main(int argc, char* argv[])
{
	string Video_Path;					// Electron tells us where the video is located...
	string Output_Path;					// ... And Electron tells us where to write the output
	double Selected_Frame_TimeStamp;	// Electron specifies which frame it wants as a timestamp in milliseconds

	if (argc != 4) {
		printf("Invalid usage: %s filename", argv[0]);
	}
	else {
		Video_Path		= argv[1];
		Output_Path		= argv[2];
		Selected_Frame_TimeStamp = atof(argv[3]);
	}

	Mat frame;
	VideoCapture cap(Video_Path);
	namedWindow("Video Capture", WINDOW_NORMAL);

	int frameCount = 0;
	double currentTime;
	while ((char)waitKey(1) != 'q')
	{
		frameCount++; 
		currentTime = timeStep*frameCount;
		double currentTime_ms = 1000 * currentTime;

		cap >> frame;
		if (frame.empty()) break;

		/* Crap Ass OpenCV library has bugs: (always returns 0) 
			double ms = cap.get(CV_CAP_PROP_POS_MSEC);
			double T = cap.get(CV_CAP_PROP_FPS);
			double F = cap.get(CV_CAP_PROP_FRAME_COUNT);
		*/

		if (round(currentTime_ms) == Selected_Frame_TimeStamp) {		// Save the frame if its timestamp matches the desired timestamp. 
			//imshow("Video Capture", frame);							// Show us the frame to be sure ;) 
			imwrite(Output_Path, frame);								// And save it 
			break;														// No need to keep going. We can modify this to save a arbitrary number of frames
		}
		
	}
	return 0;
}

