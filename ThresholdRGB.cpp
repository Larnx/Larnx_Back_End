// RedBallTracker.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdlib.h>

///////////////////////////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;
/* void on_low_r_thresh_trackbar(int, void *);
void on_high_r_thresh_trackbar(int, void *);
void on_low_g_thresh_trackbar(int, void *);
void on_high_g_thresh_trackbar(int, void *);
void on_low_b_thresh_trackbar(int, void *);
void on_high_b_thresh_trackbar(int, void *);*/
int low_r = 97, low_g = 27, low_b = 110;
int high_r = 127, high_g = 240, high_b = 117;
int main()
{
	Mat frame, frame_threshold;
	VideoCapture cap("Clip 64 AS.mp4");
	namedWindow("Video Capture", WINDOW_NORMAL);
	namedWindow("Object Detection", WINDOW_NORMAL);
	//-- Trackbars to set thresholds for RGB values
	/* createTrackbar("Low R", "Object Detection", &low_r, 255, on_low_r_thresh_trackbar);
	createTrackbar("High R", "Object Detection", &high_r, 255, on_high_r_thresh_trackbar);
	createTrackbar("Low G", "Object Detection", &low_g, 255, on_low_g_thresh_trackbar);
	createTrackbar("High G", "Object Detection", &high_g, 255, on_high_g_thresh_trackbar);
	createTrackbar("Low B", "Object Detection", &low_b, 255, on_low_b_thresh_trackbar);
	createTrackbar("High B", "Object Detection", &high_b, 255, on_high_b_thresh_trackbar);*/
	while ((char)waitKey(1) != 'q') {
		cap >> frame;
		if (frame.empty())
			break;
		//-- Detect the object based on RGB Range Values
		inRange(frame, Scalar(low_b, low_g, low_r), Scalar(high_b, high_g, high_r), frame_threshold);
		Mat resultFrame = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
		bitwise_and(frame, frame, resultFrame, frame_threshold); // bitwise to get color from mask
		//-- Show the frames
		imshow("Video Capture", frame);
		imshow("Object Detection", resultFrame); // frame_threshold
	}
	return 0;
}
/*void on_low_r_thresh_trackbar(int, void *)
{
	low_r = min(high_r - 1, low_r);
	setTrackbarPos("Low R", "Object Detection", low_r);
}
void on_high_r_thresh_trackbar(int, void *)
{
	high_r = max(high_r, low_r + 1);
	setTrackbarPos("High R", "Object Detection", high_r);
}
void on_low_g_thresh_trackbar(int, void *)
{
	low_g = min(high_g - 1, low_g);
	setTrackbarPos("Low G", "Object Detection", low_g);
}
void on_high_g_thresh_trackbar(int, void *)
{
	high_g = max(high_g, low_g + 1);
	setTrackbarPos("High G", "Object Detection", high_g);
}
void on_low_b_thresh_trackbar(int, void *)
{
	low_b = min(high_b - 1, low_b);
	setTrackbarPos("Low B", "Object Detection", low_b);
}
void on_high_b_thresh_trackbar(int, void *)
{
	high_b = max(high_b, low_b + 1);
	setTrackbarPos("High B", "Object Detection", high_b);
}*/