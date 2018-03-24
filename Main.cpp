/*
ECE Senior Design
Project Larnx

Napassorn Lerdsudwichai
Christina Howard
Kestutis Subacius
*/

#define _CRT_SECURE_NO_WARNINGS
#include<opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
#include <stdlib.h>
#include <map>

using namespace std;
using namespace cv;


void saveFrame(string Video_Path, string Output_Directory_Path, double Selected_Frame_TimeStamp, string File_Name)
{
	Mat frame;
	VideoCapture cap(Video_Path);
	string FRAME_NAME = File_Name + ".jpg";

	namedWindow("Video Capture", WINDOW_NORMAL);

	int timeStep = 1;
	int frameCount = 0;
	double currentTime;
	while ((char)waitKey(1) != 'q')
	{
		frameCount++;
		currentTime = timeStep * frameCount;
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

void readCalibration(VideoCapture capLeft, VideoCapture capRight, int num_images,
	int img_width, int img_height, char* imgsLeft_directory,
	char* imgsRight_directory, char* extension, char* stereo_calibration_filename) {

	Mat imgLeft, img_resLeft, imgRight, img_resRight;

	while ((char)waitKey(1) != 'q') {

		capLeft >> imgLeft;
		capRight >> imgRight;

		if (imgLeft.empty()) break;
		if (imgRight.empty()) break;

		resize(imgLeft, img_resLeft, Size(img_width, img_height));
		resize(imgRight, img_resRight, Size(img_width, img_height));

		imshow("IMGLeft", imgLeft);
		imshow("IMGRight", imgRight);

		if ((char)waitKey(1) == 's') {
			num_images++;
			char filenameLeft[200], filenameRight[200];
			sprintf(filenameLeft, "%s/%sleft%d.%s", stereo_calibration_filename, imgsLeft_directory, num_images, extension);
			sprintf(filenameRight, "%s/%sright%d.%s", stereo_calibration_filename, imgsRight_directory, num_images, extension);
			cout << "Saving img pair " << num_images << endl;
			imwrite(filenameLeft, img_resLeft);
			imwrite(filenameRight, img_resRight);
		}
	}
}

void setup_calibration(int board_width, int board_height, int num_imgs,
	float square_size, char* imgs_directory, char* imgs_filename, char* extension,
	Mat img, Mat gray, vector<Point2f> corners, vector<vector<Point2f>> image_points,
	vector<vector<Point3f>> object_points) {

	printf("Getting the board size\n");
	Size board_size = Size(board_width, board_height);
	int board_n = board_width * board_height;


	for (int k = 1; k <= num_imgs; k++) {

		char img_file[100];
		sprintf(img_file, "%s%s%d.%s", imgs_directory, imgs_filename, k, extension);
		img = imread(img_file, CV_LOAD_IMAGE_COLOR);
		cv::cvtColor(img, gray, CV_BGR2GRAY);
		printf("%s \n", img_file);
		bool found = false;
		found = cv::findChessboardCorners(img, board_size, corners,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		printf("Checking if found in %s \n", img_file);

		if (found)
		{
			cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
				TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(gray, board_size, corners, found);
		}

		vector< Point3f > obj;
		for (int i = 0; i < board_height; i++)
			for (int j = 0; j < board_width; j++)
				obj.push_back(Point3f((float)j * square_size, (float)i * square_size, 0));

		if (found) {
			cout << k << ". Found corners!" << endl;
			image_points.push_back(corners);
			object_points.push_back(obj);
		}
	}
}

double computeReprojectionErrors(const vector< vector< Point3f > >& objectPoints,
	const vector< vector< Point2f > >& imagePoints,
	const vector< Mat >& rvecs, const vector< Mat >& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs) {
	vector< Point2f > imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	vector< float > perViewErrors;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); ++i) {
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
			distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);
		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err / n);
		totalErr += err * err;
		totalPoints += n;
	}
	return std::sqrt(totalErr / totalPoints);
}


void intrinsicCalib(int board_width, int board_height, int num_imgs, float square_size,
	char* imgs_directory, char* imgs_filename, char* out_file, char* extension, Mat img,
	Mat gray, vector<Point2f> corners, vector<vector<Point2f>> image_points,
	vector<vector<Point3f>> object_points, char* stereo_calibration_filename) {

	printf("In the intrinsic function\n");

	setup_calibration(board_width, board_height, num_imgs, square_size, imgs_directory, imgs_filename, extension, img, gray, corners, image_points, object_points);

	printf("Starting Calibration\n");
	Mat K;
	Mat D;
	vector< Mat > rvecs, tvecs;
	int flag = 0;
	flag |= CV_CALIB_FIX_K4;
	flag |= CV_CALIB_FIX_K5;
	cout << "object_points: " << object_points.empty() << endl;
	cout << "image_points: " << image_points.empty() << endl;
	cout << "image size: " << img.size() << endl;

	calibrateCamera(object_points, image_points, img.size(), K, D, rvecs, tvecs, flag);

	cout << "Calibration error: " << computeReprojectionErrors(object_points, image_points, rvecs, tvecs, K, D) << endl;

	FileStorage fs(out_file, FileStorage::WRITE);
	fs << "K" << K;
	fs << "D" << D;
	fs << "board_width" << board_width;
	fs << "board_height" << board_height;
	fs << "square_size" << square_size;
	printf("Done Calibration\n");
}

void load_image_points(int board_width, int board_height, int num_imgs, float square_size,
	char* leftimg_dir, char* rightimg_dir, char* leftimg_filename, char* rightimg_filename,
	Mat img1, Mat img2, Mat gray1, Mat gray2, vector<Point2f> corners1, vector<Point2f> corners2,
	vector< vector< Point2f > > imagePoints1, vector< vector< Point2f > > imagePoints2,
	vector< vector< Point3f > > object_points, vector< vector< Point2f > > left_img_points,
	vector< vector< Point2f > > right_img_points, char* stereo_calibration_filename) {

	Size board_size = Size(board_width, board_height);
	int board_n = board_width * board_height;

	for (int i = 1; i <= num_imgs; i++) {
		char left_img[100], right_img[100];
		sprintf(left_img, "%s/%s%s%d.jpg", stereo_calibration_filename, leftimg_dir, leftimg_filename, i);
		sprintf(right_img, "%s/%s%s%d.jpg", stereo_calibration_filename, rightimg_dir, rightimg_filename, i);
		img1 = imread(left_img, CV_LOAD_IMAGE_COLOR);
		img2 = imread(right_img, CV_LOAD_IMAGE_COLOR);
		cvtColor(img1, gray1, CV_BGR2GRAY);
		cvtColor(img2, gray2, CV_BGR2GRAY);

		bool found1 = false, found2 = false;

		found1 = cv::findChessboardCorners(img1, board_size, corners1,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		found2 = cv::findChessboardCorners(img2, board_size, corners2,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		if (found1)
		{
			cv::cornerSubPix(gray1, corners1, cv::Size(5, 5), cv::Size(-1, -1),
				cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			cv::drawChessboardCorners(gray1, board_size, corners1, found1);
		}
		if (found2)
		{
			cv::cornerSubPix(gray2, corners2, cv::Size(5, 5), cv::Size(-1, -1),
				cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			cv::drawChessboardCorners(gray2, board_size, corners2, found2);
		}

		vector< Point3f > obj;
		for (int i = 0; i < board_height; i++)
			for (int j = 0; j < board_width; j++)
				obj.push_back(Point3f((float)j * square_size, (float)i * square_size, 0));

		if (found1 && found2) {
			cout << i << ". Found corners!" << endl;
			imagePoints1.push_back(corners1);
			imagePoints2.push_back(corners2);
			object_points.push_back(obj);
		}
	}
	for (int i = 0; i < imagePoints1.size(); i++) {
		vector< Point2f > v1, v2;
		for (int j = 0; j < imagePoints1[i].size(); j++) {
			v1.push_back(Point2f((double)imagePoints1[i][j].x, (double)imagePoints1[i][j].y));
			v2.push_back(Point2f((double)imagePoints2[i][j].x, (double)imagePoints2[i][j].y));
		}
		left_img_points.push_back(v1);
		right_img_points.push_back(v2);
	}
}


void extrinsicCalibration(char* leftcalib_file, char* rightcalib_file, char* leftimg_dir, char* rightimg_dir, char* leftimg_filename, char* rightimg_filename, char* out_file, int num_imgs, Mat img1,
	Mat img2, Mat gray1, Mat gray2, vector<Point2f> corners1, vector<Point2f> corners2,
	vector< vector< Point2f > > imagePoints1, vector< vector< Point2f > > imagePoints2,
	vector< vector< Point3f > > object_points, vector< vector< Point2f > > left_img_points,
	vector< vector< Point2f > > right_img_points, char* stereo_calibration_filename) {

	FileStorage fsl(leftcalib_file, FileStorage::READ);
	FileStorage fsr(rightcalib_file, FileStorage::READ);

	load_image_points(fsl["board_width"], fsl["board_height"], num_imgs, fsl["square_size"],
		leftimg_dir, rightimg_dir, leftimg_filename, rightimg_filename, img1, img2,
		gray1, gray2, corners1, corners2, imagePoints1, imagePoints2, object_points,
		left_img_points, right_img_points, stereo_calibration_filename);

	printf("Starting Calibration\n");
	Mat K1, K2, R, F, E;
	Vec3d T;
	Mat D1, D2;
	fsl["K"] >> K1;
	fsr["K"] >> K2;
	fsl["D"] >> D1;
	fsr["D"] >> D2;
	int flag = 0;
	flag |= CV_CALIB_FIX_INTRINSIC;

	cout << "Read intrinsics" << endl;

	stereoCalibrate(object_points, left_img_points, right_img_points, K1, D1, K2, D2, img1.size(), R, T, E, F);

	cv::FileStorage fs1(out_file, cv::FileStorage::WRITE);
	fs1 << "K1" << K1;
	fs1 << "K2" << K2;
	fs1 << "D1" << D1;
	fs1 << "D2" << D2;
	fs1 << "R" << R;
	fs1 << "T" << T;
	fs1 << "E" << E;
	fs1 << "F" << F;

	printf("Done Calibration\n");

	printf("Starting Rectification\n");

	cv::Mat R1, R2, P1, P2, Q;
	stereoRectify(K1, D1, K2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);

	fs1 << "R1" << R1;
	fs1 << "R2" << R2;
	fs1 << "P1" << P1;
	fs1 << "P2" << P2;
	fs1 << "Q" << Q;

	printf("Done Rectification\n");
}

void readRectify(VideoCapture capLeft, VideoCapture capRight, int& num_images,
	int img_width, int img_height, char* imgsLeft_directory,
	char* imgsRight_directory, char* extension) {

	Mat imgLeft, img_resLeft, imgRight, img_resRight;

	while ((char)waitKey(1) != 'q') {

		capLeft >> imgLeft;
		capRight >> imgRight;

		if (imgLeft.empty()) break;
		if (imgRight.empty()) break;

		resize(imgLeft, img_resLeft, Size(img_width, img_height));
		resize(imgRight, img_resRight, Size(img_width, img_height));

		imshow("Left Camera", imgLeft);
		imshow("Right Camera", imgRight);

		if ((char)waitKey(1) == 's') {
			num_images++;
			char filenameLeft[200], filenameRight[200];
			sprintf(filenameLeft, "%s\\left%d.%s", imgsLeft_directory, num_images, extension);
			sprintf(filenameRight, "%s\\right%d.%s", imgsRight_directory, num_images, extension);
			cout << "Saving img pair " << num_images << endl;
			imwrite(filenameLeft, img_resLeft);
			imwrite(filenameRight, img_resRight);
		}
	}
}

void undistort_rectify(int& num_imgs, string calib_file, char* left_directory, char* left_filename,
	char* right_directory, char* right_filename, char* extension) {
	// Computes the undistortion and rectification transformation map - to be called after calibration and rectification function

	Mat img1, img2, imgU1, imgU2;
	Mat R1, R2, P1, P2, Q;
	Mat K1, K2, R;
	Vec3d T;
	Mat D1, D2;

	// load calibration parameters used to calculate the rectification transform amp
	FileStorage fs(calib_file, cv::FileStorage::READ); // calib_file is a yml file stores all the calibration and rectification parameters

													   // load calibration parameters used to calculate the rectification transform amp
	fs["K1"] >> K1;
	fs["K2"] >> K2;
	fs["D1"] >> D1;
	fs["D2"] >> D2;
	fs["R"] >> R;
	fs["T"] >> T;

	fs["R1"] >> R1;
	fs["R2"] >> R2;
	fs["P1"] >> P1;
	fs["P2"] >> P2;
	fs["Q"] >> Q;

	printf("Calculating rectification transform map and remapping pixel positions\n");

	Mat lmapx, lmapy, rmapx, rmapy;
	for (int k = 1; k <= num_imgs; k++) {


		char img_file1[100], img_file2[100];
		sprintf(img_file1, "%s\\%s%d.%s", left_directory, left_filename, k, extension);
		printf("Rectifying %s \n", img_file1);
		img1 = imread(img_file1, CV_LOAD_IMAGE_COLOR);
		cvtColor(img1, img1, CV_BGR2GRAY);

		sprintf(img_file2, "%s\\%s%d.%s", right_directory, right_filename, k, extension);
		printf("Rectifying %s \n", img_file2);
		img2 = imread(img_file2, CV_LOAD_IMAGE_COLOR);
		cvtColor(img2, img2, CV_BGR2GRAY);

		Mat imgU1(img1.size(), CV_8UC1);
		Mat imgU2(img2.size(), CV_8UC1);

		// Generates a rectification transform map
		initUndistortRectifyMap(K1, D1, R1, P1, img1.size(), CV_32F, lmapx, lmapy); // left
		initUndistortRectifyMap(K2, D2, R2, P2, img2.size(), CV_32F, rmapx, rmapy); // right

																					// remapping/ relocation pixel positions in calibrated images to the rectification transform maps
		remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR); // left
		remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR); // right

		imshow("Rectified Left", imgU1);
		imshow("Rectified Right", imgU2);

		char img_outfile1[100], img_outfile2[100];
		sprintf(img_outfile1, "%s\\left_rect%d.%s", left_directory, k, extension);
		sprintf(img_outfile2, "%s\\right_rect%d.%s", right_directory, k, extension);
		printf("Saving %s \n", img_outfile1);
		printf("Saving %s \n", img_outfile2);
		imwrite(img_outfile1, imgU1);
		imwrite(img_outfile2, imgU2);

	}

}

void disparityMapping(int& num_imgs, char* left_directory, char* left_filename,
	char* right_directory, char* right_filename, char* extension, char* Output) {
	// Creates depth map from stereo videos; after calibrating and rectifying images, can determine the depth of a point in frame 

	Mat disparity, imgU1, imgU2, dis;
	// set parameters for StereoBM object - NEED TO PLAY AROUND TO FIND OPTIMUM
	int numOfDisparities = 32; // max disparities must be positive integer divisible by 16
	int blockSize = 15; // block size (window size) must be positive odd
	Ptr<StereoBM> sbm = StereoBM::create(numOfDisparities, blockSize);

	for (int k = 1; k <= num_imgs; k++) {
		char img_file1[100], img_file2[100];
		sprintf(img_file1, "%s\\%s%d.%s", left_directory, left_filename, k, extension);
		sprintf(img_file2, "%s\\%s%d.%s", right_directory, right_filename, k, extension);
		printf("Calculating disparity of %s and %s\n", img_file1, img_file2);
		imgU1 = imread(img_file1, CV_LOAD_IMAGE_COLOR);
		imgU2 = imread(img_file2, CV_LOAD_IMAGE_COLOR);
		Mat imgU1(imgU1.size(), CV_8UC1);
		Mat imgU2(imgU2.size(), CV_8UC1);

		/*sbm->setPreFilterCap(31);
		sbm->setTextureThreshold(10);
		sbm->setUniquenessRatio(15);
		sbm->setSpeckleWindowSize(100);
		sbm->setSpeckleRange(32);
		sbm->setDisp12MaxDiff(1);*/

		/*Compute the disparity for 2 rectified 8-bit single-channel frames. The disparity will be 16-bit signed
		(fixed-point) or 32-bit floating point frame of the same size as the input*/
		sbm->compute(imgU1, imgU2, disparity);
		disparity.convertTo(dis, CV_8UC1);

		char dis_file[100];
		sprintf(dis_file, "%s\\disparity%d.%s", Output, k, extension);
		printf("Saving to %s \n", dis_file);
		imwrite(dis_file, dis);
		char winame[100];
		sprintf(winame, "Disparity Map %d", k);
		imshow(winame, dis);


		// FIND DEPTH BY INVERTING disparity value???
	}
}

int main(int argc, char *argv[]) {

	// First argument determines behavior
	int method = atoi(argv[1]);

	// Initialize Variables - Not all will be used 
	string Video_Path, Video_Path1, Video_Path2, calibrated_left_path, calibrated_right_path, Frame_Path;
	string Output_Directory_Path;
	string File_Name;
	string leftout_filename, rightout_filename, calib_file;
	double Selected_Frame_TimeStamp;
	double Trim_Start, Trim_End;
	int num_imgs;
	char* left_initial_video;
	char* right_initial_video;
	char* left_image_dir;
	char* right_image_dir;
	char* left_calib_filename;
	char* right_calib_filename;
	char* stereo_calibration_filename;
	char* calib_filepath;
	char* imgs_directory;

	// Keep a dictionary of methods, for error throwing and reference. 
	std::map<int, string> first;
	first[1] = "Process Video";
	first[2] = "Save Frame";
	first[3] = "Trim Video";
	first[4] = "Track object";
	first[5] = "Histogram Analysis";
	first[6] = "Depth Mapping";
	first[7] = "Calibrate Camera";
	first[8] = "Get Chessboard Videos";

	switch (method)
	{
	case 1:	// 1: Process Video
	{
		if (argc != 5) {
			printf("Invalid usage: Method %s in process %s", first[1], argv[0]);
		}
		else {
			Video_Path = argv[2];
			Output_Directory_Path = argv[3];
			File_Name = argv[4];
			File_Name = "\\" + File_Name;
		}

		ThresholdHSV(Video_Path, Output_Directory_Path, File_Name);

		break;
	}
	case 2:	// 2: Save Selected Frame
	{
		if (argc != 6) {
			printf("Invalid usage: Method %s in process %s", first[2], argv[0]);
		}
		else {
			Video_Path = argv[2];
			Output_Directory_Path = argv[3];
			Selected_Frame_TimeStamp = atof(argv[4]);
			File_Name = argv[5];
			File_Name = "\\" + File_Name;
		}

		saveFrame(Video_Path, Output_Directory_Path, Selected_Frame_TimeStamp, File_Name);

		break;
	}
	case 3:	// 3: Trim Video
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
	case 6:		// 6. Depth Mapping calibrated and rectified images
	{
		if (argc != 8) {
			printf("Invalid usage: Method %s in process %s", first[6], argv[0]);
		}
		else {
			Video_Path1 = argv[2]; // directory of left video
			Video_Path2 = argv[3]; // directory of right video
			calib_file = argv[4]; // path and file name of calibration+ rectification parameters
			imgs_directory = argv[5]; // output video path
			left_image_dir = argv[6]; // rectified left video
			right_image_dir = argv[7]; //  rectified right video

			int num_imgs = 0;
			VideoCapture cap1(Video_Path1);
			VideoCapture cap2(Video_Path2);

			// get screenshots images for creating disparity map
			readRectify(cap1, cap2, num_imgs, cap1.get(CV_CAP_PROP_FRAME_WIDTH), cap1.get(CV_CAP_PROP_FRAME_HEIGHT), left_image_dir, right_image_dir, "jpg");

			// call rectificationfunction
			undistort_rectify(num_imgs, calib_file, left_image_dir, "left", right_image_dir, "right", "jpg");
			// returns disparity map for each frame
			disparityMapping(num_imgs, left_image_dir, "left_rect", right_image_dir, "right_rect", "jpg", imgs_directory);
		}
		break;
	}
	case 7:		// 7. Calibrate Camera
	{
		if (argc != 10) {
			printf("Invalide usage: Method %s in process %s", first[7], argv[0]);
		}
		else {
			// left calibration video
			left_initial_video = argv[2];

			// right calibration video
			right_initial_video = argv[3];

			// left image directory
			left_image_dir = argv[4];

			// right image directory
			right_image_dir = argv[5];

			// intrinsic calib output file
			left_calib_filename = argv[6];
			right_calib_filename = argv[7];

			// extrinsic Calibration
			stereo_calibration_filename = argv[8];

			// filepath to save .yml, jpg, and left & right img imgs_directory
			calib_filepath = argv[9];


			int x = 0; // num of images

			vector< vector< Point3f > > object_points;

			// intrinsic one camera calibration
			vector< vector< Point2f > > image_points;
			vector< Point2f > corners;

			Mat img, gray;
			Size im_size;

			// extrinsic two-camera calibration
			vector< vector< Point2f > > imagePoints1, imagePoints2;
			vector< Point2f > corners1, corners2;
			vector< vector< Point2f > > left_img_points, right_img_points;

			Mat img1, img2, gray1, gray2;

			VideoCapture capLeft(left_initial_video);
			VideoCapture capRight(right_initial_video);

			// get test images to calibrate
			readCalibration(capLeft, capRight, x, 1280, 720, left_image_dir, right_image_dir, "jpg", stereo_calibration_filename);

			intrinsicCalib(8, 8, x, 20.06375, left_image_dir, "left", left_calib_filename, "jpg", img, gray,
				corners, image_points, object_points, stereo_calibration_filename);

			intrinsicCalib(8, 8, x, 20.06375, right_image_dir, "right", right_calib_filename, "jpg", img, gray,
				corners, image_points, object_points, stereo_calibration_filename);

			extrinsicCalibration(left_calib_filename, right_calib_filename, left_image_dir, right_image_dir, "left", "right", stereo_calibration_filename, x, img1, img2, gray1, gray2, corners1, corners2, imagePoints1, imagePoints2, object_points,
				left_img_points, right_img_points, stereo_calibration_filename);
		}
		break;
	}
	case 8:		// 8. Get Chessboard videos
	{
		if (argc != 3) {
			printf("Invalide usage: Method %s in process %s", first[8], argv[0]);
		}
		else {

			imgs_directory = argv[2];

			//The number of connected USB camera(s)
			const uint CAM_NUM = 2;

			//This will hold the VideoCapture objects
			VideoCapture camCaptures[CAM_NUM];

			//Initialization of VideoCaptures
			for (int i = 0; i < CAM_NUM; i++)
			{

				//Opening camera capture stream
				camCaptures[i].open(i);
			}

			/*VideoCapture cap1(1);
			if (!cap1.isOpened()) cout << "Left CAM doesn't work" << endl;
			VideoCapture cap2(2);
			if (!cap2.isOpened()) cout << "Right CAM doesn't work" << endl;*/

			int key = 0;
			Mat img1, img2;
			int x = 0; // num of images saved


			while (key != 27) { // 27 = ascii value of ESC
								//cap1 >> img1;
								//cap2 >> img2;
				camCaptures[0] >> img1;
				camCaptures[1] >> img2;

				if (img1.empty()) break;
				if (img2.empty()) break;

				imshow("CAM1", img1);
				imshow("CAM2", img2);
				key = waitKey(10);
				if (key != 27) { // 27 = ascii value of ESC
					x++;
					char filename1[200], filename2[200];
					sprintf(filename1, "%s/left%d.%s", imgs_directory, x, "jpg");
					sprintf(filename2, "%s/right%d.%s", imgs_directory, x, "jpg");
					cout << "Saving img pair " << x << endl;
					imwrite(filename1, img1);
					imwrite(filename2, img2);
				}
			}
			//Releasing all VideoCapture resources
			for (int i = 0; i < CAM_NUM; i++)
			{
				camCaptures[i].release();
			}
		}

	}
	default:
		break;
	}
	return 0;
}
