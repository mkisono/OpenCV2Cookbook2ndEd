
#if !defined DTRACKER
#define DTRACKER

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudabgsegm.hpp"

#include "videoprocessor.h"

class DiceTracker : public FrameProcessor {

	cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> mog;
	std::vector<std::vector<cv::Point> > contours;

	std::vector<cv::Rect> current_rects;
	std::vector<cv::Rect> previous_rects;

	cv::Point2f dice_position[2];
	int generation[2];
	int dice_count;

	const static int ROI_SIZE = 32;
	cv::Mat dice1, dice2;

  public:

	DiceTracker() {
		mog = cv::cuda::createBackgroundSubtractorMOG2();
	}
	
	// processing method
	void process(cv:: Mat &frame, cv:: Mat &output) {

		frame.copyTo(output);

		cv::Mat foreground;
		extractObjects(frame, foreground);

		cv::imshow("fore", foreground);

		cv::findContours(foreground,contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

		int max_area = processContours();

		if (max_area > 2000 || current_rects.size() > 2) 
			return;

		// std::vector<cv::Rect>::const_iterator dice = current_rects.begin();
		// for (int i=0; dice != current_rects.end(); ++dice, i++) {

		// 	if(!isInside(frame, dice->x, dice->y))
		// 		break;

		// 	std::vector<cv::Rect>::const_iterator rect = previous_rects.begin();
		// 	for (; rect != previous_rects.end(); ++rect) {
		// 	cv::Rect intersection = *rect & *dice;
		// 	if (intersection.area() > 0) {
		// 		if (generation[i] == 0) {
		// 			dice_position[i] = cv::Point2f((dice->x + dice->width/2.0), (dice->y + dice->height/2.0));
		// 			generation[i]++;
		// 		}
		// 		else {
		// 			if (dice_position[i].inside(*dice)) {
		// 				dice_position[i] = cv::Point2f((dice->x + dice->width/2.0), (dice->y + dice->height/2.0));
		// 				generation[i]++;
		// 				std::cout << "point (" << dice_position[i].x << ", " << dice_position[i].y << ") - count: " << generation[i] << std::endl;
		// 			}
		// 			else {
		// 				generation[i] = 0;
		// 				std::cout << "reset generation[" << i << "]" << std::endl;
		// 			}
		// 		}

		// 		if (generation[i] == 10) {

		// 			cv::Mat roi = output(cv::Rect(dice_position[i].x - 16, dice_position[i].y - 16, 32, 32));

		// 			std::stringstream ss_count;
		// 			ss_count << dice_count;
		// 			std::string str = ss_count.str();
		// 			// cv::imwrite("sg" + std::to_string(dice_count) + ".jpg", roi);
		// 			cv::imwrite("sg" + str + ".jpg", roi);
		// 			dice_count++;

		// 			cv::resize(roi, roi, cv::Size(), 3.0, 3.0);
		// 			if (i == 0){
		// 				roi.copyTo(dice1);
		// 			}
		// 			else {
		// 				roi.copyTo(dice2);
		// 			}
		// 		}
		// 	}
		// }
		std::swap(current_rects, previous_rects);
	}

	int processContours() {

		int area;
		int max_area = 0;

		current_rects.clear();

		std::vector<std::vector<cv::Point> >::const_iterator contour = contours.begin();
		for(; contour != contours.end(); ++contour) {
			cv::Rect bounding_rect = cv::boundingRect(*contour);

			area = contourArea(*contour);
			if (area > max_area)
				max_area = area;
			current_rects.push_back(bounding_rect);
		}

		return max_area;
	}

	bool isInside(cv::Mat &frame, int x, int y){

		if (x < 16 ||
			y < 16 ||
			x > (frame.cols - ROI_SIZE) ||
			y > (frame.rows - ROI_SIZE)) {

			std::cout << "(" << x << ", " << y << ") - out of bounds" << std::endl;
			return false;
		}

		return true;
	}

	void extractObjects(cv::Mat &frame, cv::Mat &foreground) {

		cv::cuda::GpuMat d_frame, d_foreground;

		d_frame.upload(frame);
		mog->apply(d_frame, d_foreground, 0.003f);

		cv::cuda::threshold(d_foreground, d_foreground, 128, 255, cv::THRESH_BINARY);
		cv::Mat ker = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::Ptr<cv::cuda::Filter> erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, d_foreground.type(), ker, cv::Point(-1, -1), 1);
		cv::Ptr<cv::cuda::Filter> dilate = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, d_foreground.type(), ker, cv::Point(-1, -1), 2);
		// cv::Ptr<cv::cuda::Filter> open = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, d_foreground.type(), ker, cv::Point(-1, -1), 1);
		erode->apply(d_foreground, d_foreground);
		dilate->apply(d_foreground, d_foreground);
		// open->apply(d_foreground, d_foreground);

		d_foreground.download(foreground);
	}
};

#endif