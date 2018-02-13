/* 
	ECE Senior Design
	Project Larnx 

	Napassorn Lerdsudwichai 
	Christina Howard
	Kestutis Subacius
*/

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

Mat undistort_rectify_depth_map(Mat img1, Mat img2, FileStorage fs1) {
	// Computes the undistortion and rectification transformation map - to be called after calibration and rectification function
	// calibration function saves calibrated parameters in a file and the left and right calibrated images in a calibration directory
	// Stereo image rectification projects images onto a common image plane in such a way that the corresponding points have the same row coordinates. This image projection makes the image appear as though the two cameras are parallel.
	// rectification function calculates the rectification transform which will be used to undistort and remap the calibrated images in the rectification transform map
	// img1, img2 are the left and right calibrated images
	// variable that contains calibration parameters


	Mat R1, R2, P1, P2, Q;
	Mat K1, K2, R;
	Vec3d T;
	Mat D1, D2;

	// load calibration parameters used to calculate the rectification transform amp
	fs1["K1"] >> K1;
	fs1["K2"] >> K2;
	fs1["D1"] >> D1;
	fs1["D2"] >> D2;
	fs1["R"] >> R;
	fs1["T"] >> T;

	fs1["R1"] >> R1;
	fs1["R2"] >> R2;
	fs1["P1"] >> P1;
	fs1["P2"] >> P2;
	fs1["Q"] >> Q;

	Mat lmapx, lmapy, rmapx, rmapy;
	Mat imgU1, imgU2;

	// Generates a rectification transform map
	initUndistortRectifyMap(K1, D1, R1, P1, img1.size(), CV_32F, lmapx, lmapy); // left
	initUndistortRectifyMap(K2, D2, R2, P2, img2.size(), CV_32F, rmapx, rmapy); // right
																				// remapping/ relocation pixel positions in calibrated images to the rectification transform maps
	remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR); // left
	remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR); // right

														// calls depth/ disparity map function
	return disparityMapping(imgU1, imgU2);
}

Mat disparityMapping(Mat imgU1, Mat imgU2) {
	// Creates depth map from stereo videos; after calibrating and rectifying images, can determine the depth of a point in frame 

	Mat disparity, disparity8;

	// set parameters for StereoBM object - NEED TO PLAY AROUND TO FIND OPTIMUM
	int numOfDisparities = 16; // max disparities must be positive integer divisible by 16
	int blockSize = 9; // block size (window size) must be positive odd
	Ptr<StereoBM> sbm = StereoBM::create(numOfDisparities, blockSize);
	/*sbm->setPreFilterCap(31);
	sbm->setTextureThreshold(10);
	sbm->setUniquenessRatio(15);
	sbm->setSpeckleWindowSize(100);
	sbm->setSpeckleRange(32);
	sbm->setDisp12MaxDiff(1);*/

	/*Compute the disparity for 2 rectified 8-bit single-channel frames. The disparity will be 16-bit signed
	(fixed-point) or 32-bit floating point frame of the same size as the input*/
	sbm->compute(imgU1, imgU2, disparity);
	disparity.convertTo(disparity8, CV_8U);
	return disparity8;
}

int main(int argc, char *argv[]) {

	// First argument determines behavior
	int method = atoi(argv[1]);

	// Initialize Variables - Not all will be used 
	string Video_Path;
	string Output_Directory_Path;
	string File_Name;
	string leftout_filename, rightout_filename, calib_file;
	double Selected_Frame_TimeStamp;
	double Trim_Start, Trim_End;
	int num_imgs;

	// Keep a dictionary of methods, for error throwing and reference. 
	std::map<int, string> first;
	first[1] = "Process Video";
	first[2] = "Save Frame";
	first[3] = "Trim Video";
	first[4] = "Track object";
	first[5] = "Histogram Analysis";
	first[6] = "Depth Mapping";


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
		case 6:		// 6. Depth Mapping calibrated and rectified images
		{
			if (argc != 10) {
				printf("Invalid usage: Method %s in process %s", first[6], argv[0]);
			}
			else {
				calibrated_left_path = argv[2]; // directory that has left calibrated images - FOR ASSUME THE CHECKERBOARD IMAGES, BUT SHOULD WORK FOR THROAT TOO
				calibrated_right_path = argv[3]; // directory that has right calibrated images
				num_imgs = atoi(argv[4]); // FOR NOW, USER HAS TO INPUT TOTAL CALIBRATED IMAGES IN DIR - Christina is saving 40
				calib_file = argv[5]; // path and file name of calibration+ rectification parameters
				leftout_filename = argv[6]; // calibrated + rectified left video
				rightout_filename = argv[7]; // calibrated + rectified right video
				Output_Directory_Path = argv[8]; // disparity video path
				File_Name = argv[9]; // filename of disparity mapping video
				leftout_filename = "\\" + leftout_filename;
				rightout_filename = "\\" + rightout_filename;
				File_Name = "\\" + File_Name;

				Mat img1, img2, dis;
				Size frameSize = Size(1280, 720); // same as calibration
				int fps = 30; // assuming 30 fps

							  // load calibration parameters used to calculate the rectification transform amp
				FileStorage fs(calib_file, cv::FileStorage::READ); // calib_file is a yml file stores all the calibration and rectification parameters

																   // Write new videos
				File_Name = File_Name + ".avi"; // can change from avi
				string dis_filename = File_Name;
				leftout_filename = leftout_filename + ".avi";
				string left_calib_filename = leftout_filename;
				rightout_filename = rightout_filename + ".avi";
				string right_calib_filename = rightout_filename;

				// display window for viewing
				namedWindow("Calibrated and Rectified Sensor 0", WINDOW_NORMAL);
				namedWindow("Calibrated and Rectified Sensor 1", WINDOW_NORMAL);
				namedWindow("Disparity Map", WINDOW_NORMAL);

				// create output window
				VideoWriter writer_left, writer_right, writer_dis;
				writer_left.open(Output_Directory_Path + left_calib_filename, 0, fps, frameSize, 1);
				writer_right.open(Output_Directory_Path + right_calib_filename, 0, fps, frameSize, 1);
				writer_dis.open(Output_Directory_Path + dis_filename, 0, fps, frameSize, 1);


				for (int i = 1; i <= num_imgs; i++) {
					// read calibrated images in directory
					char left_img[100], right_img[100];
					// MAY NEED TO CHANGE FOR USER
					sprintf(left_img, "%s%s%d.jpg", calibrated_left_path, "left", i); // In Christina's code path for calibrated images is "calib_imgsLeft/1/"
					sprintf(right_img, "%s%s%d.jpg", calibrated_right_path, "right", i); // In Christina's code path for calibrated images is "calib_imgsRight/1/"
					img1 = imread(left_img, CV_LOAD_IMAGE_COLOR);
					img2 = imread(right_img, CV_LOAD_IMAGE_COLOR);

					// write to videos
					writer_left.write(img1);
					writer_right.write(img2);

					// call rectification remapping and disparity map function
					// pass left and right calibrated images and the parameters
					dis = undistort_rectify_depth_map(img1, img2, fs); // returns disparity map for each frame

																	   // write to video
					writer_dis.write(dis);

					// display videos
					imshow("Calibrated and Rectified Sensor 0", img1);
					imshow("Calibrated and Rectified Sensor 1", img2);
					imshow("Disparity Map", dis);

				}
			}
			break;
		}
		default: 
			break;
	}
	return 0;
}
