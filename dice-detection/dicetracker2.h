
#if !defined DTRACKER
#define DTRACKER

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "videoprocessor.h"

class DiceTracker : public FrameProcessor {

	cv::gpu::MOG2_GPU mog;
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
		dice_count = 0;
	}
	
	// processing method
	void process(cv:: Mat &frame, cv:: Mat &output) {

		frame.copyTo(output);

		cv::Mat foreground;
		extractObjects(frame, foreground);
		cv::imshow("Foreground", foreground);

		cv::findContours(foreground, contours,CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		int max_area = processContours();

		if (max_area < 2000 && current_rects.size() < 3) 
			detectDice(frame);

		showDice(output);

		std::swap(current_rects, previous_rects);
	}

	void detectDice(cv::Mat &frame) {

		std::vector<cv::Rect>::const_iterator dice = current_rects.begin();
		for (int i=0; dice != current_rects.end(); ++dice, i++) {

			if(!isInside(frame, dice->x, dice->y))
				break;

			std::vector<cv::Rect>::const_iterator rect = previous_rects.begin();
			for (; rect != previous_rects.end(); ++rect) {
				cv::Rect intersection = *rect & *dice;
				if (intersection.area() > 0) {
					if (generation[i] == 0) {
						dice_position[i] = cv::Point2f((dice->x + dice->width/2.0), (dice->y + dice->height/2.0));
						generation[i]++;
					}
					else {
						if (dice_position[i].inside(*dice)) {
							dice_position[i] = cv::Point2f((dice->x + dice->width/2.0), (dice->y + dice->height/2.0));
							generation[i]++;
							std::cout << "point (" << dice_position[i].x << ", " << dice_position[i].y << ") - count: " << generation[i] << std::endl;
						}
						else {
							generation[i] = 0;
							std::cout << "reset generation[" << i << "]" << std::endl;
						}
					}

					if (generation[i] == 10) {

						cv::Mat roi = frame(cv::Rect(dice_position[i].x - 16, dice_position[i].y - 16, 32, 32));

						std::stringstream ss_count;
						ss_count << dice_count;
						std::string str = ss_count.str();
						// cv::imwrite("sg" + std::to_string(dice_count) + ".jpg", roi);
						cv::imwrite("/home/mkisono/Dropbox/dice/data/mafa" + str + ".jpg", roi);
						dice_count++;

						cv::resize(roi, roi, cv::Size(), 3.0, 3.0);
						if (i == 0){
							roi.copyTo(dice1);
						}
						else {
							roi.copyTo(dice2);
						}
					}
				}
			}
		}
	}

	void showDice(cv::Mat &output) {

		int y = (output.rows / 2) - 48;
		cv::Rect dice1_position = cv::Rect(20, y, 96, 96);
		cv::Rect dice2_position = cv::Rect(140, y, 96, 96);
		cv::rectangle(output, dice1_position, cv::Scalar(128,128,128), 2);
		cv::rectangle(output, dice2_position, cv::Scalar(128,128,128), 2);

		cv::Mat dice1_show = output(dice1_position);
		cv::Mat dice2_show = output(dice2_position);

		dice1.copyTo(dice1_show);
		dice2.copyTo(dice2_show);
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

			// std::cout << "(" << x << ", " << y << ") - out of bounds" << std::endl;
			return false;
		}

		return true;
	}

	void extractObjects(cv::Mat &frame, cv::Mat &foreground) {

		cv::gpu::GpuMat d_frame, d_foreground;

		d_frame.upload(frame);
		mog.operator()(d_frame, d_foreground, 0.003f);

		cv::gpu::threshold(d_foreground, d_foreground, 128, 255, cv::THRESH_BINARY);
		cv::Mat ker = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::Ptr<cv::gpu::FilterEngine_GPU> erode = cv::gpu::createMorphologyFilter_GPU(cv::MORPH_ERODE, d_foreground.type(), ker, cv::Point(-1, -1), 1);
		cv::Ptr<cv::gpu::FilterEngine_GPU> dilate = cv::gpu::createMorphologyFilter_GPU(cv::MORPH_DILATE, d_foreground.type(), ker, cv::Point(-1, -1), 2);
		// cv::Ptr<cv::cuda::Filter> open = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, d_foreground.type(), ker, cv::Point(-1, -1), 1);
		erode->apply(d_foreground, d_foreground);
		dilate->apply(d_foreground, d_foreground);
		// open->apply(d_foreground, d_foreground);

		d_foreground.download(foreground);
	}
};

#endif
