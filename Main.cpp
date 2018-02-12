/* 
	ECE Senior Design
	Project Larnx 

	Napassorn Lerdsudwichai 
	Christina Howard
	Kestutis Subacius
*/

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

//double const	fps = 30;
//double const	timeStep = 1 / fps;


void saveFrame(string Video_Path, string Output_Directory_Path, double Selected_Frame_TimeStamp, string File_Name)
{
	Mat frame;
	VideoCapture cap(Video_Path);
	string FRAME_NAME = File_Name + ".jpg";

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
			imwrite(Output_Directory_Path + FRAME_NAME, frame);			// and save it 
			break;														// no need to keep going. we can modify this to save a arbitrary number of frames
		}
	}
}

void ThresholdHSV(string Video_Path, string Output_Directory_Path, string File_Name) {

	// get video
	Mat bright, brightHSV;
	// VideoCapture cap(Video_Path);
	VideoCapture cap(Video_Path);
	/*End Video Parameters*/

	string CSV_NAME = File_Name + ".csv";
	string MP4_NAME = File_Name + ".avi";

	namedWindow("Video Capture", WINDOW_NORMAL);
	namedWindow("Object Detection", WINDOW_NORMAL);

	/* Current Function: Segments green from video, returns pixel data vs time stamps for UI to plot to console, saves segmented vdo */
	/* Future Functions: Track green residue with bound boxes, returns ratio pixel data vs time stamp*/
	//create an ofstream for the file output 

	ofstream outputFile;
	outputFile.open(Output_Directory_Path + CSV_NAME);
	outputFile << "Frame Count" << "," << "Time (sec)" << "," << "Pixel Intensity" << endl; 	// write the file headers for csv
																							
	Size frameSize(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));		// frame width = 1280, height = 720
	int fps = cap.get(CAP_PROP_FPS);															// 30 frames per second

	VideoWriter writer;
	writer.open(Output_Directory_Path + MP4_NAME, 0, fps, frameSize, 1);

	while ((char)waitKey(1) != 'q') {
		cap >> bright; 
		if (bright.empty())
			break;

		cvtColor(bright, brightHSV, COLOR_BGR2HSV); // convert to HSV color space

		//-- Detect the object based on HSV Range Values
		//Scalar minHSV = Scalar(hsvPixel.val[0] - 40, hsvPixel.val[1] - 40, hsvPixel.val[2] - 40);
		//Scalar maxHSV = Scalar(hsvPixel.val[0] + 40, hsvPixel.val[1] + 40, hsvPixel.val[2] + 40);
		Scalar minHSV = Scalar(45, 1, 1);
		Scalar maxHSV = Scalar(123, 254, 254);

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

		//cout << (cv::sum(resultHSV)[0] / (1280 * 720 / 2)) << endl; // sort of a ratio- just dividing by frame area/2 (from observation)
		cout << (cv::sum(resultHSV)[0]) << endl;
		cvtColor(resultHSV, resultHSV, COLOR_HSV2BGR); // convert back to rgb

		writer.write(resultHSV);

		imshow("Video Capture", bright);
		imshow("Object Detection", resultHSV);

	}
	// close the output file
	outputFile.close();
}

void contourTrack(string Video_Path, string Output_Directory_Path, string File_Name) {
	// get video
	Mat bright, brightHSV;
	// VideoCapture cap(Video_Path);
	VideoCapture cap(Video_Path);
	/*End Video Parameters*/

	File_Name = File_Name + ".avi";

	namedWindow("Video Capture", WINDOW_NORMAL);
	namedWindow("Object Tracking", WINDOW_NORMAL);

	/* Current Functions: Track green residue with bound contours and boxes*/

	// Write new video
	string filenameVDO = File_Name;

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
		Scalar minHSV = Scalar(45, 1, 1);
		Scalar maxHSV = Scalar(125, 254, 254);

		Mat maskHSV, resultHSV;
		inRange(brightHSV, minHSV, maxHSV, maskHSV);
		bitwise_and(brightHSV, brightHSV, resultHSV, maskHSV);

		// Find contours
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(maskHSV, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		Mat brightClone = bright.clone();
		// Approximate Contours to resize the contours
		vector<vector<Point> > contours_poly(contours.size());
		for (int i = 0; i < contours.size(); i++) {
			approxPolyDP(Mat(contours[i]), contours_poly[i], 30, true); // Can change max distance between contours for merging
			vector<Point> hull;
			convexHull(Mat(contours_poly[i]), hull);
			Mat hull_points(hull);
			RotatedRect rotated_bounding_rect = minAreaRect(hull_points); // rotated rectangle created for each merged contour
			Point2f vertices[4];
			if (rotated_bounding_rect.size.area() == 0) {
				continue;
			}
			rotated_bounding_rect.points(vertices);
			for (int i = 0; i < 4; ++i)
			{
				line(brightClone, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 0, CV_AA); // rotated rect border is green

			}
			// Draw the bound box of each rotated rectangle
			Rect brect = rotated_bounding_rect.boundingRect();
			rectangle(brightClone, brect, Scalar(0, 0, 255), 3); // bounding box border is red

		}

		writer.write(brightClone);

		imshow("Video Capture", bright);
		imshow("Object Tracking", brightClone);
	}
}

void histogramAnalysis(string Video_Path, string Frame_Path, string Output_Directory_Path, string File_Name) {
	// For now just displays histogram for each video frame
	// get video
	Mat bright, brightHSV;
	// VideoCapture cap(Video_Path);
	VideoCapture cap(Video_Path);
	/*End Video Parameters*/

	string MP4_NAME = File_Name + ".avi";

	namedWindow("Video Capture", WINDOW_NORMAL);
	namedWindow("Histogram Analysis", WINDOW_NORMAL);

	/* Current Function: Display the Hue channel histogram of the HSV colorspace of each video frame*/
	/* Future Functions: Compare each frame histogram to that of no residue,
	then may be able to calculate the difference and thus how much alien object is in the frame
	from these histogram anomalies*/

	// ideally get frame by having user input frame chosen from save frame function
	Mat src, hsv;
	src = imread(Frame_Path);
	cvtColor(src, hsv, CV_BGR2HSV);

	// Separate the image in 3 places ( B, G and R )
	vector<Mat> img_plane;
	split(src, img_plane);

	// Intialize video frame histogram
	// Establish the number of bins
	int histSizeH = 180; // H: 0 -179
						 // int histSizeSV = 256; // S, V: 0- 255
						 // int histSize = 64; // RGB = 0 -255

						 // Set the ranges ( for H,S,V )
	float range[] = { 0, 180 }; // upper bound inclusive
								// float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false; // histogram bins will be same size and are cleared in the beginning

	Size frameSize(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT)); // frame width = 1280, height = 720
	int fps = cap.get(CAP_PROP_FPS);													// 30 frames per second

	VideoWriter writer;
	writer.open(Output_Directory_Path + MP4_NAME, 0, fps, frameSize, 1);

	while ((char)waitKey(1) != 'q') {
		cap >> bright;
		if (bright.empty())
			break;

		cvtColor(bright, brightHSV, COLOR_BGR2HSV); // convert to HSV color space

		vector<Mat> hsv_planes;						// Separate the frame in 3 places ( H, S, and V)
		split(brightHSV, hsv_planes);				// split channels of HSV
													//split(bright, bgr_planes);

		Mat h_hist, img_hist; //, s_hist, v_hist;
							  // Compute the histograms:
		calcHist(&hsv_planes[0], 1, 0, Mat(), h_hist, 1, &histSizeH, &histRange, uniform, accumulate); // dim = 1, channel = 0;
		calcHist(&img_plane[0], 1, 0, Mat(), img_hist, 1, &histSizeH, &histRange, uniform, accumulate);
		// calcHist(&bgr_planes[1], 1, 0, Mat(), s_hist, 1, &histSize, &histRange, uniform, accumulate);
		// calcHist(&bgr_planes[2], 1, 0, Mat(), v_hist, 1, &histSize, &histRange, uniform, accumulate);

		// Draw the histograms for H, S and V
		int hist_w = cap.get(CV_CAP_PROP_FRAME_WIDTH); int hist_h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		int bin_wH = cvRound((double)hist_w / histSizeH);
		// int bin_wSV = cvRound((double)hist_w / histSizeSV);
		// int bin_w = cvRound((double)hist_w / histSize);

		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

		// Normalize the result to [ 0, histImage.rows ]
		normalize(h_hist, h_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(img_hist, img_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		// normalize(s_hist, s_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		// normalize(v_hist, v_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		// output the total pixels of alien object (in green range of HSV: 45-123)
		int baseGreen = sum(img_hist(Range(45, 124), Range::all()))[0] > 0 ? sum(img_hist(Range(45, 124), Range::all()))[0] : 0;
		int alienPixels = sum(h_hist(Range(45, 124), Range::all()))[0] - baseGreen > 0 ?
			sum(h_hist(Range(45, 124), Range::all()))[0] - baseGreen : 0;
		cout << alienPixels << '\n';
		// cout << sum(h_hist(Range(45, 124), Range::all()))[0] << '\n';

		// Draw for each channel
		for (int i = 1; i < histSizeH; i++)
		{
			line(histImage, Point(bin_wH*(i - 1), hist_h - cvRound(h_hist.at<float>(i - 1))),
				Point(bin_wH*(i), hist_h - cvRound(h_hist.at<float>(i))),
				Scalar(0, 255, 0), 2, 8, 0);
			line(histImage, Point(bin_wH*(i - 1), hist_h - cvRound(img_hist.at<float>(i - 1))),
				Point(bin_wH*(i), hist_h - cvRound(img_hist.at<float>(i))),
				Scalar(0, 0, 255), 2, 8, 0);
		}

		/*for (int i = 1; i < histSizeSV; i++) {
		line(histImage, Point(bin_wSV*(i - 1), hist_h - cvRound(s_hist.at<float>(i - 1))),
		Point(bin_wSV*(i), hist_h - cvRound(s_hist.at<float>(i))),
		Scalar(0, 0, 255), 2, 8, 0);
		line(histImage, Point(bin_wSV*(i - 1), hist_h - cvRound(v_hist.at<float>(i - 1))),
		Point(bin_wSV*(i), hist_h - cvRound(v_hist.at<float>(i))),
		Scalar(255, 0, 0), 2, 8, 0);
		}*/
		// BGR
		/*for (int i = 1; i < histSize; i++) {
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(h_hist.at<float>(i - 1))),
		Point(bin_w*(i), hist_h - cvRound(h_hist.at<float>(i))),
		Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(s_hist.at<float>(i - 1))),
		Point(bin_w*(i), hist_h - cvRound(s_hist.at<float>(i))),
		Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(v_hist.at<float>(i - 1))),
		Point(bin_w*(i), hist_h - cvRound(v_hist.at<float>(i))),
		Scalar(0, 0, 255), 2, 8, 0);
		}*/


		writer.write(histImage);

		imshow("Video Capture", bright);
		imshow("Histogram Analysis", histImage);

	}



}

int main(int argc, char *argv[]) {

	// First argument determines behavior
	int method = atoi(argv[1]);

	// Initialize Variables - Not all will be used 
	string Video_Path;
	string Output_Directory_Path;
	string File_Name;
	double Selected_Frame_TimeStamp;
	double Trim_Start, Trim_End;

	// Keep a dictionary of methods, for error throwing and reference. 
	std::map<int, string> first;
	first[1] = "Process Video";
	first[2] = "Save Frame";
	first[3] = "Trim Video";
	first[4] = "Track object";
	first[5] = "Histogram Analysis";


	switch (method) 
	{
		case 1 :	// 1: Process Video
		{
			if (argc != 5) {
				printf("Invalid usage: Method %s in process %s", first[1], argv[0]);
			}
			else {
				Video_Path				= argv[2];
				Output_Directory_Path	= argv[3];
				File_Name				= argv[4];
				File_Name = "\\" + File_Name;
			}

			ThresholdHSV(Video_Path, Output_Directory_Path, File_Name);

			break;
		}
		case 2 :	// 2: Save Selected Frame
		{
			if (argc != 6) {
				printf("Invalid usage: Method %s in process %s", first[2], argv[0]);
			}
			else {
				Video_Path				 = argv[2];
				Output_Directory_Path	 = argv[3];
				Selected_Frame_TimeStamp = atof(argv[4]);
				File_Name				 = argv[5];
				File_Name = "\\" + File_Name;
			}

			saveFrame(Video_Path, Output_Directory_Path, Selected_Frame_TimeStamp, File_Name);

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
		case 4:    // 4. Track Object
		{
			if (argc != 5) {
				printf("Invalid usage: Method %s in process %s", first[4], argv[0]);
			}
			else {
				Video_Path = argv[2];
				Output_Directory_Path = argv[3];
				File_Name = argv[4];
				File_Name = "\\" + File_Name;
			}

			contourTrack(Video_Path, Output_Directory_Path, File_Name);

			break;
		}
		case 5:		// 5. Histogram Analysis
		{
			if (argc != 6) {
				printf("Invalid usage: Method %s in process %s", first[5], argv[0]);
			}
			else {
				Video_Path = argv[2];
				Frame_Path = argv[3];
				Output_Directory_Path = argv[4];
				File_Name = argv[5];
				File_Name = "\\" + File_Name;
			}

			histogramAnalysis(Video_Path, Frame_Path, Output_Directory_Path, File_Name);
			break;
		}
		default: 
			break;
	}
	return 0;
}
