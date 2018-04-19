/*
ECE Senior Design
Project Larnx
Napassorn Lerdsudwichai
Christina Howard
Kestutis Subacius
*/

#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv_modules.hpp>
// #include <opencv2/viz/viz.hpp> // need to link if use
#include <iostream>
#include <fstream>
#include <cstring>
#include <stdlib.h>
#include <thread>
#include <Windows.h>


using namespace cv;

void saveFrame(std::string Video_Path, std::string Output_Directory_Path, double Selected_Frame_TimeStamp, std::string File_Name)
{
	Mat frame;
	VideoCapture cap(Video_Path);
	std::string FRAME_NAME = File_Name + ".jpg";

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
			cv::imshow("video capture", frame);							    // show us the frame to be sure ;) 
			cv::imwrite(Output_Directory_Path + FRAME_NAME, frame);			// and save it 
			break;														// no need to keep going. we can modify this to save a arbitrary number of frames
		}
	}
}

void ThresholdHSV(std::string Video_Path, std::string Output_Directory_Path, std::string File_Name) {

	// get video
	Mat bright, brightHSV;
	// VideoCapture cap(Video_Path);
	VideoCapture cap(Video_Path);
	/*End Video Parameters*/

	std::string CSV_NAME = File_Name + ".csv";
	std::string MP4_NAME = File_Name + ".avi";

	namedWindow("Video Capture", WINDOW_NORMAL);
	namedWindow("Object Detection", WINDOW_NORMAL);

	/* Current Function: Segments green from video, returns pixel data vs time stamps for UI to plot to console, saves segmented vdo */
	/* Future Functions: Track green residue with bound boxes, returns ratio pixel data vs time stamp*/
	//create an ofstream for the file output 

	std::ofstream outputFile;
	outputFile.open(Output_Directory_Path + CSV_NAME);
	outputFile << "Frame Count" << "," << "Time (sec)" << "," << "Pixel Intensity\n"; 	// write the file headers for csv

	Size frameSize(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));		// frame width = 1280, height = 720
	int fps = cap.get(CAP_PROP_FPS);															// 30 frames per second

	VideoWriter writer;
	writer.open(Output_Directory_Path + MP4_NAME, 0, fps, frameSize, 1);

	//-- Trackbars to set thresholds for RGB values
	//createTrackbar("Low Hue", "Object Detection", &low_h, 255, on_low_h_thresh_trackbar);
	//createTrackbar("High Hue", "Object Detection", &high_h, 255, on_high_h_thresh_trackbar);

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

		Mat maskHSV, resultHSV(brightHSV.size(), CV_8UC3);
		//inRange(brightHSV, Scalar(low_h, 1, 254), Scalar(high_h, 1, 254), maskHSV);
		inRange(brightHSV, minHSV, maxHSV, maskHSV);
		bitwise_and(brightHSV, brightHSV, resultHSV, maskHSV);

		// get time stamps and sum pixels
		outputFile << cap.get(CAP_PROP_POS_FRAMES) << "," << cap.get(CAP_PROP_POS_MSEC) / 1000 << "," << cv::sum(resultHSV)[0] << std::endl;
		// print to console
		//cout << cap.get(CAP_PROP_POS_MSEC) / 1000 << endl << cv::sum(resultHSV)[0] << endl;
		//cout << "\{ \"x:\"\"" << cap.get(CAP_PROP_POS_MSEC) / 1000 << "\",\"y:\"" << cv::sum(resultHSV)[0] << "\"\}" << endl;
		// JSON Format
		//'{ "name":"John", "age":30, "city":"New York"}'

		//cout << (cv::sum(resultHSV)[0] / (1280 * 720 / 2)) << endl; // sort of a ratio- just dividing by frame area/2 (from observation)
		/*
		for (int i = 0; i < maskHSV.size().height; i++) { // Mat - 3 channel Mat<Vec3d> --> Mat<Vec3d>.at(i,j)[0] => hue
			for (int j = 0; j < maskHSV.size().width; j++) {
				if (maskHSV.at<Vec3b>(i, j)[0] > 0) {
					cout << "x: " << j << " , y: " << i << endl;
				}
			}
		}
		*/

		// cout << resultHSV << endl;
		cvtColor(resultHSV, resultHSV, COLOR_HSV2BGR); // convert back to rgb

		writer.write(resultHSV);

		cv::imshow("Video Capture", bright);
		cv::imshow("Object Detection", resultHSV);

	}
	outputFile.close();
}

void contourTrack(std::string Video_Path, std::string Output_Directory_Path, std::string File_Name) {
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
	std::string filenameVDO = File_Name;

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
		std::vector<std::vector<Point> > contours;
		std::vector<Vec4i> hierarchy;
		findContours(maskHSV, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		Mat brightClone = bright.clone();

		// Approximate Contours to resize the contours
		std::vector<std::vector<Point> > contours_poly(contours.size());
		for (int i = 0; i < contours.size(); i++) {
			approxPolyDP(Mat(contours[i]), contours_poly[i], 30, true); // Can change max distance between contours for merging
			std::vector<Point> hull;
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

		cv::imshow("Video Capture", bright);
		cv::imshow("Object Tracking", brightClone);
	}
}

void histogramAnalysis(std::string Video_Path, std::string Frame_Path, std::string Output_Directory_Path, std::string File_Name) {
	// For now just displays histogram for each video frame
	// get video
	Mat bright, brightHSV;
	// VideoCapture cap(Video_Path);
	VideoCapture cap(Video_Path);
	/*End Video Parameters*/

	std::string MP4_NAME = File_Name + ".avi";

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
	std::vector<Mat> img_plane;
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

		std::vector<Mat> hsv_planes;						// Separate the frame in 3 places ( H, S, and V)
		split(brightHSV, hsv_planes);				// split channels of HSV
													//split(bright, bgr_planes);

		Mat h_hist, img_hist; //, s_hist, v_hist;
							  // Compute the histograms:
		calcHist(&hsv_planes[0], 1, 0, Mat(), h_hist, 1, &histSizeH, &histRange, uniform, accumulate); // dim = 1, channel = 0;
		calcHist(&img_plane[0], 1, 0, Mat(), img_hist, 1, &histSizeH, &histRange, uniform, accumulate);

		// Draw the histograms for H, S and V
		int hist_w = cap.get(CV_CAP_PROP_FRAME_WIDTH); int hist_h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		int bin_wH = cvRound((double)hist_w / histSizeH);

		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

		// Normalize the result to [ 0, histImage.rows ]
		normalize(h_hist, h_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(img_hist, img_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		// output the total pixels of alien object (in green range of HSV: 45-123)
		int baseGreen = sum(img_hist(Range(45, 124), Range::all()))[0] > 0 ? sum(img_hist(Range(45, 124), Range::all()))[0] : 0;
		int alienPixels = sum(h_hist(Range(45, 124), Range::all()))[0] - baseGreen > 0 ?
			sum(h_hist(Range(45, 124), Range::all()))[0] - baseGreen : 0;
		std::cout << alienPixels << '\n';

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
		writer.write(histImage);

		imshow("Video Capture", bright);
		imshow("Histogram Analysis", histImage);
	}
}

void readCalibration(VideoCapture capLeft, VideoCapture capRight, int& num_images,
	int img_width, int img_height, std::string imgsLeft_directory,
	std::string imgsRight_directory, std::string extension) {

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
		std::string stereo_calibration_filename = "calib";
		if ((char)waitKey(1) == 's') {
			num_images++;
			std::string filenameLeft = imgsLeft_directory + "\\" + stereo_calibration_filename + std::to_string(num_images) + "." + extension;
			std::string filenameRight = imgsRight_directory + "\\" + stereo_calibration_filename + std::to_string(num_images) + "." + extension;
			std::cout << "Saving img pair " << num_images << "\n";
			imwrite(filenameLeft, img_resLeft);
			imwrite(filenameRight, img_resRight);
		}
	}
}

void setup_calibration(int board_width, int board_height, int num_imgs,
	float square_size, std::string imgs_directory, std::string imgs_filename, std::string extension,
	Mat &img, Mat& gray, std::vector<Point2f> &corners, std::vector<std::vector<Point2f>> &image_points,
	std::vector<std::vector<Point3f>> &object_points) {

	printf("Getting the board size\n");
	Size board_size = Size(board_width, board_height);
	int board_n = board_width * board_height;

	for (int k = 1; k <= num_imgs; k++) {

		char img_file[100];
		sprintf(img_file, "%s\\%s%d.%s", imgs_directory, imgs_filename, k, extension);

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

		std::vector< Point3f > obj;
		for (int i = 0; i < board_height; i++)
			for (int j = 0; j < board_width; j++)
				obj.push_back(Point3f((float)j * square_size, (float)i * square_size, 0));

		if (found) {
			std::cout << k << ". Found corners!\n";
			image_points.push_back(corners);
			object_points.push_back(obj);
		}
	}
}

double computeReprojectionErrors(const std::vector< std::vector< Point3f > >& objectPoints,
	const std::vector< std::vector< Point2f > >& imagePoints,
	const std::vector< Mat >& rvecs, const std::vector< Mat >& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs) {
	// NOT used in fisheye
	std::vector< Point2f > imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	std::vector< float > perViewErrors;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); ++i) {
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
		// cv::fisheye::projectPoints(Mat(objectPoints[i]), imagePoints2, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, 0.0, noArray());
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);
		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err / n);
		totalErr += err * err;
		totalPoints += n;
	}
	return std::sqrt(totalErr / totalPoints);
}

void intrinsicCalib(int board_width, int board_height, int num_imgs, float square_size, const char* imgs_directory, const char* imgs_filename,
	const char* out_file, const char* extension, Mat img, Mat gray, std::vector<Point2f> corners, std::vector<std::vector<Point2f>> image_points,
	std::vector<std::vector<Point3f>> object_pointsI) {

	printf("In the intrinsic function\n");

	setup_calibration(board_width, board_height, num_imgs, square_size, imgs_directory, imgs_filename, extension, img, gray, corners, image_points, object_pointsI);

	printf("Starting Calibration\n");
	Mat K;
	Mat D;
	std::vector< Mat > rvecs, tvecs;
	int flag = 0;
	//flag |= CV_CALIB_SAME_FOCAL_LENGTH;
	//flag |= CV_CALIB_FIX_PRINCIPAL_POINT;
	//flag |= CV_CALIB_FIX_ASPECT_RATIO;
	//flag |= CV_CALIB_ZERO_TANGENT_DIST;
	//flag |= CV_CALIB_RATIONAL_MODEL;
	//flag |= CV_CALIB_FIX_K3;
	flag |= CV_CALIB_FIX_K4;
	flag |= CV_CALIB_FIX_K5;
	//flag |= CV_CALIB_FIX_K6;
	std::cout << "object_points: " << object_pointsI.empty() << std::endl;
	std::cout << "image_points: " << image_points.empty() << std::endl;
	std::cout << "image size: " << img.size() << std::endl;

	calibrateCamera(object_pointsI, image_points, img.size(), K, D, rvecs, tvecs, flag);

	std::cout << "Calibration error: " << computeReprojectionErrors(object_pointsI, image_points, rvecs, tvecs, K, D) << std::endl;

	FileStorage fs(out_file, FileStorage::WRITE);
	fs << "K" << K; // camera matrix
	fs << "D" << D; // distortion coeffs
	fs << "board_width" << board_width;
	fs << "board_height" << board_height;
	fs << "square_size" << square_size;
	printf("Done Calibration\n");
}

void load_image_points(int board_width, int board_height, int num_imgs, float square_size,
	const char* leftimg_dir, const char* rightimg_dir, Mat& img1, Mat img2, Mat gray1, Mat gray2, std::vector<Point2f> corners1, std::vector<Point2f> corners2,
	std::vector< std::vector< Point2f > > imagePoints1, std::vector< std::vector< Point2f > > imagePoints2,
	std::vector< std::vector< Point3f > > &object_points, std::vector< std::vector< Point2f > > &left_img_points,
	std::vector< std::vector< Point2f > > &right_img_points) {

	Size board_size = Size(board_width, board_height);
	int board_n = board_width * board_height;

	for (int i = 1; i <= num_imgs; i++) {
		char left_img[100], right_img[100];

		sprintf(left_img, "%s\\%s%d.jpg", leftimg_dir, "left", i);
		sprintf(right_img, "%s\\%s%d.jpg", rightimg_dir, "right", i);
		img1 = imread(left_img, CV_LOAD_IMAGE_COLOR);
		img2 = imread(right_img, CV_LOAD_IMAGE_COLOR);
		std::cout << "Image Size: " << img1.size() << std::endl;
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

		std::vector< Point3f > obj;
		for (int i = 0; i < board_height; i++)
			for (int j = 0; j < board_width; j++)
				obj.push_back(Point3f((float)j * square_size, (float)i * square_size, 0));

		if (found1 && found2) {
			std::cout << i << ". Found corners!\n";
			imagePoints1.push_back(corners1);
			imagePoints2.push_back(corners2);
			object_points.push_back(obj);
		}
	}
	for (int i = 0; i < imagePoints1.size(); i++) {
		std::vector< Point2f > v1, v2;
		for (int j = 0; j < imagePoints1[i].size(); j++) {
			v1.push_back(Point2f((double)imagePoints1[i][j].x, (double)imagePoints1[i][j].y));
			v2.push_back(Point2f((double)imagePoints2[i][j].x, (double)imagePoints2[i][j].y));
		}
		left_img_points.push_back(v1);
		right_img_points.push_back(v2);
	}
}


void extrinsicCalibration(const char* leftcalib_file, const char* rightcalib_file, const char* leftimg_dir, const char* rightimg_dir, const char* out_file, int num_imgs, Mat img1, Mat img2, Mat gray1, Mat gray2, std::vector<Point2f> corners1, std::vector<Point2f> corners2,
	std::vector< std::vector< Point2f > > imagePoints1, std::vector< std::vector< Point2f > > imagePoints2,
	std::vector< std::vector< Point3f > > object_pointsE, std::vector< std::vector< Point2f > > left_img_points,
	std::vector< std::vector< Point2f > > right_img_points) {

	FileStorage fsl(leftcalib_file, FileStorage::READ);
	FileStorage fsr(rightcalib_file, FileStorage::READ);

	load_image_points(fsl["board_width"], fsl["board_height"], num_imgs, fsl["square_size"], leftimg_dir, rightimg_dir,
		img1, img2, gray1, gray2, corners1, corners2, imagePoints1, imagePoints2, object_pointsE, left_img_points, right_img_points);

	printf("Starting Calibration\n");
	Mat K1 = Mat(3, 3, CV_64FC1);
	Mat K2 = Mat(3, 3, CV_64FC1);
	Vec3d T;
	Mat D1, D2;
	Mat R, F, E;
	fsl["K"] >> K1;
	fsr["K"] >> K2;
	fsl["D"] >> D1;
	fsr["D"] >> D2;
	int flag = 0;
	//flag |= CV_CALIB_FIX_INTRINSIC;
	//flag |= CV_CALIB_USE_INTRINSIC_GUESS;
	flag |= CV_CALIB_SAME_FOCAL_LENGTH;
	flag |= CV_CALIB_ZERO_TANGENT_DIST;
	//flag |= CV_CALIB_RATIONAL_MODEL;
	//flag |= CV_CALIB_FIX_K3;
	//flag |= CV_CALIB_FIX_K4;
	//flag |= CV_CALIB_FIX_K5;
	//flag |= CV_CALIB_FIX_K6;

	std::cout << "Read intrinsics\n";
	//cout << img1.size() << endl;
	double rms = stereoCalibrate(object_pointsE, left_img_points, right_img_points, K1, D1, K2, D2, img1.size(), R, T, E, F, flag,
		cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5));
	//options for term criteria --  for the iterative optimization algorithm
	// cvTermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6)
	// cvTermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-6));
	// cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5));
	std::cout << "Stereo Calibrate rms error: " << rms << "\n";

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
	int flag2 = 0;
	flag2 |= CV_CALIB_ZERO_DISPARITY;
	stereoRectify(K1, D1, K2, D2, img1.size(), R, T, R1, R2, P1, P2, Q, flag2, 0.5, img1.size());

	fs1 << "R1" << R1;
	fs1 << "R2" << R2;
	fs1 << "P1" << P1;
	fs1 << "P2" << P2;
	fs1 << "Q" << Q;

	printf("Done Rectification\n");
}

// Reproject image to 3D
void customReproject(const cv::Mat& disparity, const cv::Mat& Q, cv::Mat& out3D)
{
	CV_Assert(disparity.type() == CV_32F && !disparity.empty());
	CV_Assert(Q.type() == CV_32F && Q.cols == 4 && Q.rows == 4);

	// 3-channel matrix for containing the reprojected 3D world coordinates
	out3D = cv::Mat::zeros(disparity.size(), CV_32FC3);

	// Getting the interesting parameters from Q, everything else is zero or one
	float Q03 = Q.at<float>(0, 3);
	float Q13 = Q.at<float>(1, 3);
	float Q23 = Q.at<float>(2, 3);
	float Q32 = Q.at<float>(3, 2);
	float Q33 = Q.at<float>(3, 3);

	// Transforming a single-channel disparity map to a 3-channel image representing a 3D surface
	for (int i = 0; i < disparity.rows; i++)
	{
		const float* disp_ptr = disparity.ptr<float>(i);
		cv::Vec3f* out3D_ptr = out3D.ptr<cv::Vec3f>(i);

		for (int j = 0; j < disparity.cols; j++)
		{
			const float pw = 1.0f / (disp_ptr[j] * Q32 + Q33);

			cv::Vec3f& point = out3D_ptr[j];
			point[0] = (static_cast<float>(j) + Q03) * pw;
			point[1] = (static_cast<float>(i) + Q13) * pw;
			point[2] = Q23 * pw;
		}
	}
}


void readRectify(VideoCapture capLeft, VideoCapture capRight, int& num_images,
	int img_width, int img_height, std::string imgsLeft_directory, std::string imgsRight_directory, std::string extension, int sampleRate) {

	Mat imgLeft, img_resLeft, imgRight, img_resRight;
	int nTh_frame = sampleRate;
	int frameCount = 0;
	while ((char)waitKey(5) != 'q') { // press "q" key to escape
									  //waitKey(3);
		capLeft >> imgLeft;
		capRight >> imgRight;
		frameCount++;

		if (imgLeft.empty()) { std::cout << "Empty frame \n"; break; }
		if (imgRight.empty()) { std::cout << "Empty frame \n"; break; }

		resize(imgLeft, img_resLeft, Size(img_width, img_height));
		resize(imgRight, img_resRight, Size(img_width, img_height));

		imshow("Left Camera", imgLeft);
		imshow("Right Camera", imgRight);

		if (nTh_frame == frameCount) {
			frameCount = 0;
			num_images++;

			std::string filenameLeft = imgsLeft_directory + "\\left" + std::to_string(num_images) + "." + extension;
			std::string filenameRight = imgsRight_directory + "\\right" + std::to_string(num_images) + "." + extension;

			std::cout << "Saving img pair " << num_images << std::endl;
			imwrite(filenameLeft, img_resLeft);
			std::cout << filenameLeft << "\n";
			imwrite(filenameRight, img_resRight);
		}
	}
	capLeft.release();
	capRight.release();
	destroyAllWindows;
}


void StereoVision(int& num_imgs, std::string calib_file, const char* left_directory, const char* left_filename,
	const char* right_directory, const char* right_filename, const char* extension, const char* Output, int numOfDisparities, int blockSize, int minDisparity) {

	/* After CALIBRATION*/

	// calibration parameters
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
	fs.release();

	printf("Calculating rectification transform map and remapping pixel positions\n");

	Mat lmapx, lmapy, rmapx, rmapy;
	char img_file1[100], img_file2[100];
	Mat img1, img2;
	char img_outfile1[100], img_outfile2[100];
	char dis_file[100];
	char out_file[100];
	char out_file2[100];
	std::ofstream outdata;
	char outdata_file[100];
	/* For rectifying with color images*/
	vector<Mat> imgC1(3), imgC2(3);

	for (int k = 1; k <= num_imgs; k++) {
		// --------------------------------------------------------------------------------------
		/* -- STEREO RECTIFICATION: Computes the UNDISTORTION and RECTIFICATION TRANSFORM MAP -- */
		// --------------------------------------------------------------------------------------
		sprintf(img_file1, "%s\\%s%d.%s", left_directory, left_filename, k, extension);
		printf("Rectifying %s \n", img_file1);
		img1 = imread(img_file1, CV_LOAD_IMAGE_COLOR);
		/*Split channel to rectify each color channel*/
		split(img1, imgC1); 
		cvtColor(img1, img1, CV_BGR2GRAY); // make single channel;

		sprintf(img_file2, "%s\\%s%d.%s", right_directory, right_filename, k, extension);
		printf("Rectifying %s \n", img_file2);
		img2 = imread(img_file2, CV_LOAD_IMAGE_COLOR);
		/*Split channel to rectify each color channel*/
		split(img2, imgC2);
		cvtColor(img2, img2, CV_BGR2GRAY);


		// rectified left and right image variables
		Mat imgU1; // (img1.size(), CV_8UC1);
		Mat imgU2; // (img2.size(), CV_8UC1);

				   // Generates a rectification transform map
		initUndistortRectifyMap(K1, D1, R1, P1, img1.size(), CV_32FC1, lmapx, lmapy); // left
		initUndistortRectifyMap(K2, D2, R2, P2, img2.size(), CV_32FC1, rmapx, rmapy); // right

																					  // remapping/ relocation pixel positions in calibrated images to the rectification transform maps
		remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR); // cv::BORDER_CONSTANT, cv::Scalar());
		remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR); // cv::BORDER_CONSTANT, cv::Scalar());

		/* remap each color channel BGR */
		// left
		remap(imgC1[0], imgC1[0], lmapx, lmapy, cv::INTER_LINEAR);
		remap(imgC1[1], imgC1[1], lmapx, lmapy, cv::INTER_LINEAR);
		remap(imgC1[2], imgC1[2], lmapx, lmapy, cv::INTER_LINEAR);
		Mat imgRectColor1;
		merge(imgC1, imgRectColor1); // merge
		// right
		remap(imgC2[0], imgC2[0], lmapx, lmapy, cv::INTER_LINEAR);
		remap(imgC2[1], imgC2[1], lmapx, lmapy, cv::INTER_LINEAR);
		remap(imgC2[2], imgC2[2], lmapx, lmapy, cv::INTER_LINEAR);
		Mat imgRectColor2;
		merge(imgC2, imgRectColor2); // merge
		/* use imgRectColor1 and 2 for contour tracking */

		sprintf(img_outfile1, "%s\\left_rect_%d.%s", left_directory, k, extension);
		sprintf(img_outfile2, "%s\\right_rect_%d.%s", right_directory, k, extension);
		printf("Saving %s \n", img_outfile1);
		printf("Saving %s \n", img_outfile2);
		imwrite(img_outfile1, imgU1);
		imwrite(img_outfile2, imgU2);

		// --------------------------------------------------------------------------------------
		/* -- DISPARITY MAP GENERATION -- */
		// --------------------------------------------------------------------------------------

		Ptr<StereoBM> sbm = StereoBM::create(numOfDisparities, blockSize);
		sbm->setMinDisparity(minDisparity);
		//sbm->setROI1();
		//sbm->setPreFilterSize(5); // preFilterCap, preFilterSize, preFilterType - used in filtering the input images before disparity computation. These may improve noise rejection in input images.
		sbm->setPreFilterCap(10);
		sbm->setTextureThreshold(15); // textureThreshold, uniquenessRatio - used in filtering the disparity map before returning. May reduce noise.
		sbm->setUniquenessRatio(10);
		sbm->setSpeckleWindowSize(100); //disp12MaxDiff, speckleRange, speckleWindowSize - used in filtering the disparity map before returning, looking for areas of similar disparity (small areas will be assumed to be noise and marked as having invalid depth information). These reduces noise in disparity map output.
		sbm->setSpeckleRange(32);
		sbm->setDisp12MaxDiff(1);

		printf("Computing disparity for rectified image pair %d\n", k);
		/*Compute the disparity for 2 rectified 8-bit single-channel frames. The disparity will be 16-bit signed
		(fixed-point) or 32-bit floating point frame of the same size as the input*/
		Mat disparity;// (imgU1.size(), CV_8UC1);
		sbm->compute(imgU1, imgU2, disparity);
		Mat disp8;// (disparity.size(), CV_8UC1);
				  // normalize(disparity, disp8, 0, 255, CV_MINMAX, CV_8UC1);
				  // Check its extreme values
		double minVal; double maxVal;
		minMaxLoc(disparity, &minVal, &maxVal);
		printf("Min disparity found: %f Max disparity found: %f \n", minVal, maxVal);
		// Normalize it as a CV_8UC1 image
		disparity.convertTo(disp8, CV_8UC1, 255 / (maxVal - minVal));

		// To better visualize the result, apply a colormap to the computed disparity
		Mat cm_disp;
		applyColorMap(disp8, cm_disp, COLORMAP_JET);
		//imshow("cm disparity m", cm_disp);

		sprintf(dis_file, "%s\\disparity_%d.%s", Output, k, extension);
		printf("Saving to %s \n", dis_file);
		//imwrite(dis_file, disparity);
		imwrite(dis_file, cm_disp);

		// --------------------------------------------------------------------------------------
		/* -- POINT CLOUD OR 3D REPROJECTION OF DISPARITY MAPS -- */
		// --------------------------------------------------------------------------------------

		printf("Computing depth from disparity map %d\n", k);
		// 3D image from disparity
		Mat depth, customDepth;//  (disparity.size(), CV_32F);
		Mat disp16;
		disparity.convertTo(disp16, CV_32F, 1.0 / 16.0);
		//disparity.convertTo(disparity, CV_32F, 1.0 / 16.0);

		// Compute point cloud - reprojection of disparity map to 3D
		Q.convertTo(Q, CV_32F);
		reprojectImageTo3D(disp16, depth, Q, true, CV_32F); // Q = Output  4 \times 4 disparity-to-depth mapping matrix 
															//reprojectImageTo3D(disparity, depth, Q, true, CV_32F);
															//Q.convertTo(Q, CV_32F);
															//Mat_<float> vec(4, 1);
															// cout << "Channel: " << disp16.channels() << " Size: " << disp16.size() << endl;
															// cout << "Channel: " << depth.channels() << " Size: " << depth.size() << endl;


															//customReproject(disp16, Q, customDepth);

															//cout << "depth map size " << depth.channels() << endl;
															// every pixel will have 3D coordinates, can be obtained:
		sprintf(outdata_file, "%s\\depth%d.csv", Output, k);
		outdata.open(outdata_file);
		outdata << "Depth Map Frame " << k << std::endl;
		outdata << "Pixel X, Pixel Y, Real X, Real Y, Real Z" << std::endl;
		for (int x = 0; x < depth.cols; x++) {
			for (int y = 0; y < depth.rows; y++) {
				//vec(0) = x;
				//vec(1) = y;
				//vec(2) = disp16.at<float>(y, x);
				//vec(3) = 1;
				//vec = Q * vec;
				//vec /= vec(3);
				Point3f p = depth.at<Point3f>(y, x); // depth is p.z
													 //Point2f p2 = disp16.at<Point1f>
													 //Point3f p2 = customDepth.at<Point3f>(y, x);
													 //if (abs(p.x) > 10 || abs(p.y) > 10 || abs(p.z) > 10) { // Discard points that are too far from the camera, and thus are highly unreliable
													 //if (abs(vec(0))>10 || abs(vec(1))>10 || abs(vec(2))>10) {
													 /*if (p.z >= 10000) {
													 outdata << "too far" << "," << "too far" << "," << "too far"  << endl;
													 }*/
													 //else {
				if (p.z < 10000) {
					outdata << x << "," << y << "," << p.x << "," << p.y << "," << p.z << std::endl;
					// outdata << p.x << "," << p.y << "," << p.z << "," << endl;
					//outdata << vec(0) << "," << vec(1) << "," << vec(2) << "," << endl;
				}
				// printf("Pixel coordinates: %f %f , Depth: %f \n", p.x, p.y, p.z);  // or print to a file	 
			}
		}
		outdata.close();
		sprintf(out_file, "%s\\depth3D_%d.%s", Output, k, extension);
		printf("Saving to %s \n", out_file);
		imwrite(out_file, depth);
	}
}


void volumeCalculation(std::string Video_Path_Left, std::string Video_Path_Right, std::string Output_Directory_Path, std::string Output_Name, std::string Calibration_File, int numOfDisparities, int blockSize, int minDisparity) {

	Mat bright, brightHSV;
	Mat originalFrame_Left, originalFrame_Right;
	Mat cvtFrame_Left, cvtFrame_Right;

	VideoCapture leftCapture(Video_Path_Left);
	VideoCapture rightCapture(Video_Path_Right);
	Output_Name = Output_Name + ".avi";


	Size frameSize(leftCapture.get(CV_CAP_PROP_FRAME_WIDTH), leftCapture.get(CV_CAP_PROP_FRAME_HEIGHT));
	int  fps = leftCapture.get(CAP_PROP_FPS);

	VideoWriter leftWriter;
	leftWriter.open(Output_Directory_Path + Output_Name, 0, fps, frameSize, 1);

	while ((char)waitKey(1) != 'q')
	{
		leftCapture >> originalFrame_Left;
		rightCapture >> originalFrame_Right;
		if (originalFrame_Left.empty() || originalFrame_Right.empty()) break;

		cvtColor(originalFrame_Left, cvtFrame_Left, COLOR_BGR2HSV);
		cvtColor(originalFrame_Right, cvtFrame_Right, COLOR_BGR2HSV);

		/* Separate by Hues */
		int sensitivity = 25;
		Scalar minHSV = Scalar(60 - sensitivity, 50, 2);
		Scalar maxHSV = Scalar(60 + sensitivity, 255, 255);
		Mat hsvMask_Left, hsvResult_Left;
		Mat hsvMask_Right, hsvResult_Right;
		inRange(cvtFrame_Left, minHSV, maxHSV, hsvMask_Left);
		bitwise_and(cvtFrame_Left, cvtFrame_Left, hsvResult_Left, hsvMask_Left);
		inRange(cvtFrame_Right, minHSV, maxHSV, hsvMask_Right);
		bitwise_and(cvtFrame_Right, cvtFrame_Right, hsvResult_Right, hsvMask_Right);

		namedWindow("HSV", WINDOW_NORMAL);
		cv::imshow("HSV", hsvResult_Left);

		/* Separate by Colors */
		/*
		Scalar minRGB = Scalar(36, 0, 0);
		Scalar maxRGB = Scalar(105, 255, 255);
		Mat rgbMask_Left,  rgbResult_Left;
		Mat rgbMask_Right, rgbResult_Right;
		inRange(originalFrame_Left, minRGB, maxRGB, rgbMask_Left);
		bitwise_and(originalFrame_Left, originalFrame_Left, rgbResult_Left, rgbMask_Left);
		inRange(originalFrame_Right, minRGB, maxRGB, rgbMask_Right);
		bitwise_and(originalFrame_Right, originalFrame_Right, rgbResult_Right, rgbMask_Right);
		*/

		std::vector<std::vector<Point>> contours;
		std::vector<Vec4i> hierarchy;
		findContours(hsvMask_Left, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		//findContours(cvtFrame_Left, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		Mat cvtFrameClone_Left = cvtFrame_Left.clone();

		std::vector<std::vector<Point> >	contours_poly(contours.size());
		std::vector<Rect>					boundRect(contours.size());
		std::vector<Point2f>				center(contours.size());
		std::vector<float>					radius(contours.size());
		drawContours(cvtFrameClone_Left, contours, -1, (0, 255, 0), 3);

		Rect brect;
		RotatedRect rotated_bounding_rect;
		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			std::vector<Point> hull;
			convexHull(Mat(contours_poly[i]), hull);
			Mat hull_points(hull);
			rotated_bounding_rect = minAreaRect(hull_points); // rotated rectangle created for each merged contour
			Point2f vertices[4];
			if (rotated_bounding_rect.size.area() == 0) {
				continue;
			}
			rotated_bounding_rect.points(vertices);
			for (int i = 0; i < 4; ++i)
			{
				line(cvtFrameClone_Left, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 0, CV_AA); // rotated rect border is green
			}
			// Draw the bound box of each rotated rectangle
			brect = rotated_bounding_rect.boundingRect();
			rectangle(cvtFrameClone_Left, brect, Scalar(0, 0, 255), 3); // bounding box border is red
		}

		namedWindow("Tracked", WINDOW_NORMAL);
		cv::imshow("Tracked", cvtFrameClone_Left);

		Mat R1, R2, P1, P2, Q;
		Mat K1, K2, R;
		Vec3d T;
		Mat D1, D2;
		FileStorage fs(Calibration_File, cv::FileStorage::READ);
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
		fs.release();

		Mat lmapx, lmapy, rmapx, rmapy;
		char img_file1[100], img_file2[100];
		Mat img1, img2;
		char img_outfile1[100], img_outfile2[100];
		char dis_file[100];
		char out_file[100];
		char out_file2[100];
		char outdata_file[100];

		img1 = originalFrame_Left;
		img2 = originalFrame_Right;

		cvtColor(img1, img1, CV_BGR2GRAY); 
		cvtColor(img2, img2, CV_BGR2GRAY);

		Mat imgU1; 
		Mat imgU2; 

		initUndistortRectifyMap(K1, D1, R1, P1, img1.size(), CV_32FC1, lmapx, lmapy); // left
		initUndistortRectifyMap(K2, D2, R2, P2, img2.size(), CV_32FC1, rmapx, rmapy); // right
		remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
		remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);
		
		Ptr<StereoBM> sbm = StereoBM::create(numOfDisparities, blockSize);
		sbm->setMinDisparity(minDisparity);
		sbm->setPreFilterCap(10);
		sbm->setTextureThreshold(15);	// Used in filtering the disparity map before returning. May reduce noise.
		sbm->setUniquenessRatio(10);
		sbm->setSpeckleWindowSize(100); // Used in filtering the disparity map before returning, looking for areas of similar disparity (small areas will be assumed to be noise and marked as having invalid depth information). These reduces noise in disparity map output.
		sbm->setSpeckleRange(32);
		sbm->setDisp12MaxDiff(1);

		Mat disparity;
		sbm->compute(imgU1, imgU2, disparity);
		Mat disp8;
		double minVal; double maxVal;
		minMaxLoc(disparity, &minVal, &maxVal);
		//printf("Min disparity found: %f Max disparity found: %f \n", minVal, maxVal);

		disparity.convertTo(disp8, CV_8UC1, 255 / (maxVal - minVal));

		Mat cm_disp;
		applyColorMap(disp8, cm_disp, COLORMAP_JET);

		Mat depth, customDepth, pointCloud;
		Mat disp16;
		disparity.convertTo(disp16, CV_32F, 1.0 / 16.0);

		namedWindow("Disparity", WINDOW_NORMAL);
		cv::imshow("Disparity", cm_disp);

		Q.convertTo(Q, CV_32F);
		reprojectImageTo3D(disp16, depth, Q, true, CV_32F);


		int leftBound = brect.x;
		int rightBound = brect.x + brect.width;
		int topBound = brect.y;
		int bottomBound = brect.y - brect.height;
		float Volume = 0;
		float deltaVolume = 0;
		int outBox = 0;
		float outCount = 0;
		
		float lastX, nextX, lastY, nextY;
		lastY = depth.at<Point3f>(0, 0).y;
		nextY = depth.at<Point3f>(1, 0).y;
		lastX = depth.at<Point3f>(0, 0).x;
		nextX = depth.at<Point3f>(0, 1).x;
		float deltaX = nextX - lastX;
		float deltaY = nextY - lastY;

		for (int y = 0; y < depth.rows; y++) {
			for (int x = 0; x < depth.cols; x++) {
				Point3f p = depth.at<Point3f>(y,x);
				/*
				Point2f q = depth.at<Point2f>(y, x);
				if (pointPolygonTest(contours , q, false) >= 0) {
				deltaVolume = (p.z - avgBackground);
				deltaVolume = deltaVolume * deltaX * deltaY;
				Volume = Volume + deltaVolume;
				}
				*/
				if ((x > leftBound && x < rightBound) && (y > bottomBound && y < topBound) && (p.z < 10000)) {}
				else if (p.z < 10000) {
					outBox++;
					outCount = outCount + p.z;
				}
			}
		}
		int avgBackground = outCount / outBox;
		for (int y = 0; y < depth.rows; y++) {
			for (int x = 0; x < depth.cols; x++) {
				Point3f p = depth.at<Point3f>(y,x); 

				deltaX = depth.at<Point3f>(y, x).x - depth.at<Point3f>(y - 1, x).x;
				deltaY = depth.at<Point3f>(y, x).y - depth.at<Point3f>(y, x - 1).y;

				if ((x > leftBound && x < rightBound) && (y > bottomBound && y < topBound) && (p.z < 10000)) {
					/*
					Point2f q = depth.at<Point2f>(y, x);
					if (pointPolygonTest(contours , q, false) >= 0) {
						deltaVolume = (p.z - avgBackground);
						deltaVolume = deltaVolume * deltaX * deltaY;
						Volume = Volume + deltaVolume;
					}
					*/
					deltaVolume = p.z;//(avgBackground - p.z);
					deltaVolume = deltaVolume * deltaX * deltaY;
					Volume = Volume + deltaVolume;
				}
			}
		}
		Volume = -Volume;
		std::cout << "Volume in frame: " << Volume << "\n";
	}
}


int main(int argc, char *argv[])
{
	enum METHOD {
		HSV_Threshold_2D,	/* 0 */
		HSV_Threshold_3D,	/* 1 */
		Save_Frame_2D,		/* 2 */
		Save_Frame_3D,		/* 3 */
		Trim_Frame_2D,		/* 4 */
		Trim_Frame_3D,		/* 5 */
		Track_Object_2D,	/* 6 */
		Track_Object_3D,	/* 7 */
		Histogram_2D,		/* 8 */
		Histogram_3D,		/* 9 */
		Depth_Map,			/* 10 */
		Calibrate,			/* 11 */
		Get_Checkerboard,	/* 12 */
		Get_Camera			/* 13 */
	};

	int			method = atoi(argv[1]);
	std::string inputPath_Main = argv[2];
	std::string inputPath_Left = argv[3];
	std::string inputPath_Right = argv[4];
	std::string inputPath_Confg = argv[5];
	std::string outputName = argv[6];
	std::string outputPath = argv[7];
	int			timeStamp = atoi(argv[8]);
	int			numDisparity = atoi(argv[9]);
	int			minDisparity = atoi(argv[10]);
	int			blockSize = atoi(argv[11]);
	int			arbitraryParam = atoi(argv[12]);

	std::string outputName_Left = outputName + "_Left";
	std::string outputName_Right = outputName + "_Right";

	switch (method)
	{
	case HSV_Threshold_2D:
	{
		outputName = "\\" + outputName;
		ThresholdHSV(inputPath_Main, outputPath, outputName);
		break;
	}
	case HSV_Threshold_3D:
	{
		outputName_Left = "\\" + outputName_Left;
		outputName_Right = "\\" + outputName_Right;
		std::thread left(ThresholdHSV, inputPath_Left, outputPath, outputName_Left);
		std::thread right(ThresholdHSV, inputPath_Right, outputPath, outputName_Right);
		left.join();
		right.join();
		break;
	}
	case Save_Frame_2D:
	{
		saveFrame(inputPath_Main, outputPath, timeStamp, outputName);
		break;
	}
	case Save_Frame_3D:
	{
		std::thread left(saveFrame, inputPath_Left, outputPath, timeStamp, outputName_Left);
		std::thread right(saveFrame, inputPath_Right, outputPath, timeStamp, outputName_Right);
		left.join();
		right.join();
		break;
	}
	case Trim_Frame_2D:
	{
		break;
	}
	case Trim_Frame_3D:
	{
		break;
	}
	case Track_Object_2D:
	{
		contourTrack(inputPath_Main, outputPath, outputName);
	}
	case Track_Object_3D:
	{
		std::thread left(contourTrack, inputPath_Left, outputPath, outputName_Left);
		std::thread right(contourTrack, inputPath_Right, outputPath, outputName_Right);
		left.join();
		right.join();
		break;
	}
	case Histogram_2D:
	{
		//histogramAnalysis(inputPath_Main, Frame_Path, Output_Directory_Path, File_Name);
		break;
	}
	case Histogram_3D:
	{
		break;
	}
	case Depth_Map:
	{
		int num_imgs = 0;

		std::string outputFolder = outputPath + "\\out";
		std::string outLeft = outputFolder + "\\Left";
		std::string outRight = outputFolder + "\\Right";
		CreateDirectory(outputFolder.c_str(), NULL);
		CreateDirectory(outLeft.c_str(), NULL);
		CreateDirectory(outRight.c_str(), NULL);

		VideoCapture leftCapture(inputPath_Left);
		VideoCapture rightCapture(inputPath_Right);

		readRectify(leftCapture, rightCapture, num_imgs, leftCapture.get(CV_CAP_PROP_FRAME_WIDTH), leftCapture.get(CV_CAP_PROP_FRAME_HEIGHT), outLeft.c_str(), outRight.c_str(), "jpg", arbitraryParam);
		StereoVision(num_imgs, inputPath_Confg, outLeft.c_str(), "left", outRight.c_str(), "right", "jpg", outputFolder.c_str(), numDisparity, blockSize, minDisparity);
		break;
	}
	case Calibrate:
	{
		int x = 0;
		std::vector< std::vector< Point3f > > object_pointsI;
		std::vector< std::vector< Point3f > > object_pointsE;
		std::vector< std::vector< Point2f > > image_points;
		std::vector< std::vector< Point2d > > img_points;
		std::vector< Point2f >			corners;
		Mat img, gray;
		Size im_size;
		std::vector< std::vector< Point2f > > imagePoints1, imagePoints2;
		std::vector< Point2f >			corners1, corners2;
		std::vector< std::vector< Point2f > > left_img_points, right_img_points;
		Mat img1, img2, gray1, gray2;

		std::string outputFolder = outputPath + "\\out";
		std::string outLeft = outputFolder + "\\Left";
		std::string outRight = outputFolder + "\\Right";
		CreateDirectory(outputFolder.c_str(), NULL);
		CreateDirectory(outLeft.c_str(), NULL);
		CreateDirectory(outRight.c_str(), NULL);
		std::string leftIntrinsic = "leftIntrinsic";
		std::string rightIntrinsic = "rightIntrinsic";

		VideoCapture leftCapture(inputPath_Left);
		VideoCapture rightCapture(inputPath_Right);

		// get test images to calibrate
		readCalibration(leftCapture, rightCapture, x, 319, 240, outLeft.c_str(), outRight.c_str(), "jpg");

		intrinsicCalib(7, 9, x, 0.024, outLeft.c_str(), "left", leftIntrinsic.c_str(), "jpg", img, gray,
			corners, image_points, object_pointsI);

		intrinsicCalib(7, 9, x, 0.024, outRight.c_str(), "right", rightIntrinsic.c_str(), "jpg", img, gray,
			corners, image_points, object_pointsI);

		extrinsicCalibration(leftIntrinsic.c_str(), rightIntrinsic.c_str(), outLeft.c_str(), outRight.c_str(), outputPath.c_str(), x, img1, img2, gray1, gray2, corners1, corners2, imagePoints1, imagePoints2, object_pointsE, left_img_points, right_img_points);

		break;
	}
	case Get_Checkerboard:
	{
		VideoCapture cap1(0);
		if (!cap1.isOpened()) std::cout << "Left CAM doesn't work" << std::endl;
		VideoCapture cap2(1);
		if (!cap2.isOpened()) std::cout << "Right CAM doesn't work" << std::endl;

		int key = 0;
		Mat img1, img2;
		int x = 0; // num of images saved

		while (key != 27) { // 27 = ascii value of ESC
			cap1 >> img1;
			cap2 >> img2;


			if (img1.empty()) break;
			if (img2.empty()) break;

			imshow("CAM1", img1);
			imshow("CAM2", img2);
			key = cvWaitKey(10);
			if (key != 27) { // 27 = ascii value of ESC
				x++;
				char filename1[200], filename2[200];
				sprintf(filename1, "%s/left%d.%s", inputPath_Main, x, "jpg");
				sprintf(filename2, "%s/right%d.%s", inputPath_Main, x, "jpg");
				std::cout << "Saving img pair " << x << std::endl;
				imwrite(filename1, img1);
				imwrite(filename2, img2);
			}
		}
	}
	case Get_Camera:
	{
		CvCapture* captureL = 0;
		captureL = cvCreateCameraCapture(0);
		if (!captureL) {
			return -1;
		}

		CvCapture* captureR = 0;
		captureR = cvCreateCameraCapture(1);
		if (!captureR) {
			return -1;
		}

		IplImage *frameL = cvQueryFrame(captureL);//Init the video read
		IplImage *frameR = cvQueryFrame(captureR);//Init the video read
		double fps = cvGetCaptureProperty(
			captureL,
			CV_CAP_PROP_FPS
		);

		CvSize size = cvSize(
			(int)cvGetCaptureProperty(captureL, CV_CAP_PROP_FRAME_WIDTH),
			(int)cvGetCaptureProperty(captureL, CV_CAP_PROP_FRAME_HEIGHT)
		);

		CvVideoWriter *writerL = cvCreateVideoWriter(inputPath_Main.c_str(), CV_FOURCC('I', 'P', 'D', 'V'), fps, size); // AVI codec
		CvVideoWriter *writerR = cvCreateVideoWriter(inputPath_Main.c_str(), CV_FOURCC('I', 'P', 'D', 'V'), fps, size);

		while ((frameL = cvQueryFrame(captureL)) != NULL && (frameR = cvQueryFrame(captureR))) {
			cvWriteFrame(writerL, frameL);
			cvWriteFrame(writerR, frameR);

			if (cvWaitKey(-1) == 27) {
				cvReleaseVideoWriter(&writerL);
				cvReleaseVideoWriter(&writerR);
				cvReleaseCapture(&captureL);
				cvReleaseCapture(&captureR);
				break;
			}

		}
	}
	case 14:
		outputName_Left = "\\" + outputName_Left;
		outputName_Right = "\\" + outputName_Right;
		volumeCalculation(inputPath_Left, inputPath_Right, outputPath, outputName_Left, inputPath_Confg, numDisparity, blockSize, minDisparity);
	default:
		break;
	}
	return 0;
}
