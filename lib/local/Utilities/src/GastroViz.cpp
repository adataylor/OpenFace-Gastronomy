///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis, all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////

#include "GastroViz.h"
#include "VisualizationUtils.h"
#include "RotationHelpers.h"
#include "ImageManipulationHelpers.h"

#include <sstream>
#include <iomanip>
#include <map>
#include <set>

// For plotting data
#include <opencv2/plot.hpp>

// For drawing on images
// #include <opencv2/imgproc.hpp>

using namespace Utilities;

// For subpixel accuracy drawing
const int draw_shiftbits = 4;
const int draw_multiplier = 1 << 4;

const std::map<std::string, std::string> AUS_DESCRIPTION = {
	{ "AU01", "Inner Brow Raiser   " },
	{ "AU02", "Outer Brow Raiser   " },
	{ "AU04", "Brow Lowerer        " },
	{ "AU05", "Upper Lid Raiser    " },
	{ "AU06", "Cheek Raiser        " },
	{ "AU07", "Lid Tightener       " },
	{ "AU09", "Nose Wrinkler       " },
	{ "AU10", "Upper Lip Raiser    " },
	{ "AU12", "Lip Corner Puller   " },
	{ "AU14", "Dimpler             " },
	{ "AU15", "Lip Corner Depressor" },
	{ "AU17", "Chin Raiser         " },
	{ "AU20", "Lip stretcher       " },
	{ "AU23", "Lip Tightener       " },
	{ "AU25", "Lips part           " },
	{ "AU26", "Jaw Drop            " },
	{ "AU28", "Lip Suck            " },
	{ "AU45", "Blink               " },
};

cv::Scalar color_rose = cv::Scalar(116, 77, 152);
cv::Scalar color_pink = cv::Scalar(111, 0, 217);
cv::Scalar color_red = cv::Scalar(53, 67, 234);
cv::Scalar color_orange = cv::Scalar(0, 99, 215);
cv::Scalar color_yellow = cv::Scalar(5, 188, 251);
cv::Scalar color_green = cv::Scalar(83, 168, 52);
cv::Scalar color_blue = cv::Scalar(244, 133, 66);
cv::Scalar color_lavender = cv::Scalar(152, 77, 116);
cv::Scalar color_purple = cv::Scalar(217, 0, 111);

cv::Scalar color_level_none = cv::Scalar(0, 0, 0, 255);
cv::Scalar color_level_0 = cv::Scalar(0, 191, 63, 255);
cv::Scalar color_level_1 = cv::Scalar(0, 191, 141, 255);
cv::Scalar color_level_2 = cv::Scalar(0, 162, 191, 255);
cv::Scalar color_level_3 = cv::Scalar(0, 85, 191, 255);
cv::Scalar color_level_4 = cv::Scalar(0, 7, 191, 255);

// TODO check interrupt color levels
// currently not used
// interrupt relies on the same levels as need (ie color_level)

cv::Scalar interrupt_color_level_none = cv::Scalar(0, 0, 100);
cv::Scalar interrupt_color_level_0 = cv::Scalar(63, 191, 75);
cv::Scalar interrupt_color_level_1 = cv::Scalar(141, 191, 75);
cv::Scalar interrupt_color_level_2 = cv::Scalar(191, 162, 75);
cv::Scalar interrupt_color_level_3 = cv::Scalar(191, 85, 75);
cv::Scalar interrupt_color_level_4 = cv::Scalar(191, 7, 75);

cv::Scalar person_color_0 = color_purple;
cv::Scalar person_color_1 = color_orange;
cv::Scalar person_color_2 = color_blue;
cv::Scalar person_color_3 = color_green;
cv::Scalar person_color_4 = color_pink;
cv::Scalar person_color_5 = color_red;

const int AU_TRACKBAR_LENGTH = 400;
const int AU_TRACKBAR_HEIGHT = 10;

const int MARGIN_X = 185;
const int MARGIN_Y = 10;

const int window_size = 50;
const int average_size = 3;

float needLog_0[window_size];
float interruptLog_0[window_size];
float needLogSmooth_0[window_size];
float interruptLogSmooth_0[window_size];

float needLog_1[window_size];
float interruptLog_1[window_size];
float needLogSmooth_1[window_size];
float interruptLogSmooth_1[window_size];

float needLog_2[window_size];
float interruptLog_2[window_size];
float needLogSmooth_2[window_size];
float interruptLogSmooth_2[window_size];

float needLog_3[window_size];
float interruptLog_3[window_size];
float needLogSmooth_3[window_size];
float interruptLogSmooth_3[window_size];

float needLog_4[window_size];
float interruptLog_4[window_size];
float needLogSmooth_4[window_size];
float interruptLogSmooth_4[window_size];

float needLog_5[window_size];
float interruptLog_5[window_size];
float needLogSmooth_5[window_size];
float interruptLogSmooth_5[window_size];

float needLog_6[window_size];
float interruptLog_6[window_size];
float needLogSmooth_6[window_size];
float interruptLogSmooth_6[window_size];


// Set up the GastroViz instance
GastroViz::GastroViz(std::vector<std::string> arguments)
{
	// By default not visualizing anything
	// FALSE, by default we are visualizing AUs
	// TODO check if also visualizing some other things	
	this->vis_track = true;
	this->vis_hog = false;
	this->vis_align = false;
	this->vis_aus = true;

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-verbose") == 0)
		{
			this->vis_track = true;
			this->vis_align = true;
			this->vis_hog = true;
			this->vis_aus = true;
		}
		else if (arguments[i].compare("-vis-align") == 0)
		{
			this->vis_align = true;
		}
		else if (arguments[i].compare("-vis-hog") == 0)
		{
			this->vis_hog = true;
		}
		else if (arguments[i].compare("-vis-track") == 0)
		{
			this->vis_track = true;
		}
		else if (arguments[i].compare("-vis-aus") == 0)
		{
			this->vis_aus = true;
		}
	}

}

GastroViz::GastroViz(bool vis_track, bool vis_hog, bool vis_align, bool vis_aus)
{
	this->vis_track = vis_track;
	this->vis_hog = vis_hog;
	this->vis_align = vis_align;
	this->vis_aus = vis_aus;
}

// Setting the image on which to draw
void GastroViz::SetImage(const cv::Mat& canvas, float fx, float fy, float cx, float cy)
{
	// Convert the image to 8 bit RGB
	captured_image = canvas.clone();

	this->fx = fx;
	this->fy = fy;
	this->cx = cx;
	this->cy = cy;

	// Clearing other images
	hog_image = cv::Mat();
	aligned_face_image = cv::Mat();
	action_units_image = cv::Mat();

	classifier_image = cv::Mat();
	//top_view_image = cv::Mat();
	graph_image = cv::Mat();
}


void GastroViz::SetObservationFaceAlign(const cv::Mat& aligned_face)
{
	if(this->aligned_face_image.empty())
	{
		this->aligned_face_image = aligned_face;
	}
	else
	{
		cv::vconcat(this->aligned_face_image, aligned_face, this->aligned_face_image);
	}
}

void GastroViz::SetObservationHOG(const cv::Mat_<double>& hog_descriptor, int num_cols, int num_rows)
{
	if(vis_hog)
	{
		if (this->hog_image.empty())
		{
			Visualise_FHOG(hog_descriptor, num_rows, num_cols, this->hog_image);
		}
		else
		{
			cv::Mat tmp_hog;
			Visualise_FHOG(hog_descriptor, num_rows, num_cols, tmp_hog);
			cv::vconcat(this->hog_image, tmp_hog, this->hog_image);
		}
	}

}

// Draws the dots on the faces of the person if they have been successfully located
void GastroViz::SetObservationLandmarks(const cv::Mat_<float>& landmarks_2D, double confidence, const cv::Mat_<int>& visibilities)
{

	if(confidence > visualisation_boundary)
	{
		// Draw 2D landmarks on the image
		int n = landmarks_2D.rows / 2;

		// Drawing feature points
		for (int i = 0; i < n; ++i)
		{
			if (visibilities.empty() || visibilities.at<int>(i))
			{
				cv::Point featurePoint(cvRound(landmarks_2D.at<float>(i) * (float)draw_multiplier), cvRound(landmarks_2D.at<float>(i + n) * (float)draw_multiplier));

				// A rough heuristic for drawn point size
				int thickness = (int)std::ceil(3.0* ((double)captured_image.cols) / 640.0);
				int thickness_2 = (int)std::ceil(1.0* ((double)captured_image.cols) / 640.0);

				cv::circle(captured_image, featurePoint, 1 * draw_multiplier, cv::Scalar(0, 0, 255), thickness, cv::LINE_AA, draw_shiftbits);
				cv::circle(captured_image, featurePoint, 1 * draw_multiplier, cv::Scalar(255, 0, 0), thickness_2, cv::LINE_AA, draw_shiftbits);

			}
			else
			{
				// Draw a fainter point if the landmark is self occluded
				cv::Point featurePoint(cvRound(landmarks_2D.at<float>(i) * (double)draw_multiplier), cvRound(landmarks_2D.at<float>(i + n) * (double)draw_multiplier));

				// A rough heuristic for drawn point size
				int thickness = (int)std::ceil(2.5* ((double)captured_image.cols) / 640.0);
				int thickness_2 = (int)std::ceil(1.0* ((double)captured_image.cols) / 640.0);

				cv::circle(captured_image, featurePoint, 1 * draw_multiplier, cv::Scalar(0, 0, 155), thickness, cv::LINE_AA, draw_shiftbits);
				cv::circle(captured_image, featurePoint, 1 * draw_multiplier, cv::Scalar(155, 0, 0), thickness_2, cv::LINE_AA, draw_shiftbits);

			}
		}
	}
}


// Draw the box around the person's face if it's sufficiently confident
void GastroViz::SetObservationPose(int personId, const cv::Vec6f& pose, double confidence)
{

	cv::Scalar person_color = cv::Scalar(255, 255, 255);

	if (personId == 0) {
		person_color = person_color_0;		
	} else if (personId == 1) {
		person_color = person_color_1;		
	} else if (personId == 2) {
		person_color = person_color_2;		
	} else if (personId == 3) {
		person_color = person_color_3;		
	} else if (personId == 4) {
		person_color = person_color_4;		
	} else if (personId == 5) {
		person_color = person_color_5;
	} 


	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (confidence > visualisation_boundary)
	{
		double vis_certainty = confidence;
		if (vis_certainty > 1)
			vis_certainty = 1;

		// Scale from 0 to 1, to allow to indicated by colour how confident we are in the tracking
		vis_certainty = (vis_certainty - visualisation_boundary) / (1 - visualisation_boundary);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		// Draw it in reddish if uncertain, blueish if certain
		DrawBox(captured_image, pose, person_color, thickness, fx, fy, cx, cy);
	}
}


void GastroViz::SetFeatures(const std::vector<std::pair<std::string, double> >& au_intensities,
	const std::vector<std::pair<std::string, double> >& au_occurences)
{
	


	
}

void GastroViz::SetNeediness(const std::vector<std::pair<std::string, double> >& au_intensities,
	const std::vector<std::pair<std::string, double> >& au_occurences)
{

//Radius point should be proportional to head size
cv::Scalar pointColor0 = cv::Scalar(0, 0, 255);
cv::Scalar pointColor1 = cv::Scalar(0, 0, 255);
cv::Scalar pointColor2 = cv::Scalar(0, 255, 255);
cv::Scalar pointColor3 = cv::Scalar(255, 0, 255);
cv::Scalar pointColor4 = cv::Scalar(0, 255, 0);
cv::Scalar pointColor5 = cv::Scalar(255, 0, 0);

cv::Scalar indicatorColor = pointColor3;

//Draw an indicator circle
// TODO turn into Sims-style indicator?
// circle(captured_image, humanPoint, radiusPoint, pointColor, thickness=1, cv::LINE_AA, shift=0);
	

}

void GastroViz::SetTopView(const cv::Vec6f& pose, double confidence, const std::vector<std::pair<std::string, double> >& au_intensities,
	const std::vector<std::pair<std::string, double> >& au_occurences)
{
	// if (top_view_image.empty())
	// {
	// 	top_view_image = cv::Mat(240, 320, CV_8UC3, cv::Scalar(255, 255, 255));
	// }

	// DISTANCES are in mm
	// Use 1/10 of a meter incremements?

	// PINK
	cv::Scalar need1 = cv::Scalar(255, 0, 255);
	// RED
	cv::Scalar need2 = cv::Scalar(0, 0, 255);
	// BLUE
 	cv::Scalar need3 = cv::Scalar(255, 0, 0);


// 	float pX = 1000.0 * pose[0] / 1000.0;
// 	float pY = 1000.0 * pose[1] / 1000.0;
// 	float pZ = 1000.0 * pose[2] / 1000.0;

// 	// Add padding to keep stuff on screen
// 	pX = pX + 40;
// 	pY = pY + 40;
// 	pZ = pZ + 40;

	// cv::Point humanPoint1 = cv::Point((int)pX, (int)pY);
	// cv::circle(top_view_image, humanPoint1, 10, need1, 5, cv::LINE_AA);

	// cv::Point humanPoint2 = cv::Point((int)fx, (int)fy);
	// cv::circle(top_view_image, humanPoint2, 10, need2, 5, cv::LINE_AA);

	//cv::Point humanPoint3 = cv::Point((int)(cx/2.0 + pose[0]), (int)(cy/2.0));
	//cv::circle(top_view_image, humanPoint3, 10, need3, 5, cv::LINE_AA);

}

void GastroViz::ClearClassifier(int numPeople)
{
	
	if (classifier_image.empty())
	{
		//To do, handle if multiple people
		classifier_image = cv::Mat(2 * numPeople * (AU_TRACKBAR_HEIGHT + 10) + MARGIN_Y * 2, AU_TRACKBAR_LENGTH + MARGIN_X, CV_8UC3, cv::Scalar(255, 255, 255));
	}
	else
	{
		classifier_image.setTo(255);
	}

}


void GastroViz::SetClassifier(bool newSet, int personId, int numPeople, const cv::Vec6f& pose, double confidence, const std::vector<std::pair<std::string, double> >& au_intensities,
	const std::vector<std::pair<std::string, double> >& au_occurences)
{
	cv::Scalar person_color;
	
	if (personId == 0) {
		person_color = person_color_0;		
	} else if (personId == 1) {
		person_color = person_color_1;		
	} else if (personId == 2) {
		person_color = person_color_2;		
	} else if (personId == 3) {
		person_color = person_color_3;		
	} else if (personId == 4) {
		person_color = person_color_4;		
	} else {
		person_color = person_color_5;		
	} 

	const int AU_TRACKBAR_LENGTH = 400;
	const int AU_TRACKBAR_HEIGHT = 10;

	const int MARGIN_X = 185;
	const int MARGIN_Y = 10;

	std::map<std::string, std::pair<bool, double>> aus;

	std::set<std::string> au_names;
	std::map<std::string, bool> occurences_map;
	std::map<std::string, double> intensities_map;

	const int nb_aus = (int) au_names.size();

	for (size_t idx = 0; idx < au_intensities.size(); idx++)
	{
		au_names.insert(au_intensities[idx].first);
		intensities_map[au_intensities[idx].first] = au_intensities[idx].second;
	}

	for (size_t idx = 0; idx < au_occurences.size(); idx++)
	{
		au_names.insert(au_occurences[idx].first);
		occurences_map[au_occurences[idx].first] = au_occurences[idx].second > 0;
	}

	double neediness = 0;
	// double interruptibility = 0;

	int numberOfMetrics = 2;
	int numberOfSlots = numberOfMetrics + 1;

	if (classifier_image.empty())
	{
		//To do, handle if multiple people
		classifier_image = cv::Mat(numberOfSlots * numPeople * (AU_TRACKBAR_HEIGHT + 10) + MARGIN_Y * 2, AU_TRACKBAR_LENGTH + MARGIN_X + AU_TRACKBAR_HEIGHT, CV_8UC3, cv::Scalar(255, 255, 255));
	}
	else if (newSet)
	{
		classifier_image.setTo(255);
	}

	if (au_intensities.size() <= 0 || au_occurences.size() <= 0)
	{
		return;
	}

	std::map<std::string, std::pair<bool, double>> classifications;

	neediness = 0;
	for (auto au_name : au_names)
	{
		// Insert the intensity and AU presense (as these do not always overlap check if they exist first)
		bool occurence = false;
		if (occurences_map.find(au_name) != occurences_map.end())
		{
			occurence = occurences_map[au_name] != 0;
		}
		else
		{
			// If we do not have an occurence label, trust the intensity one
			occurence = intensities_map[au_name] > 1;
		}
		double intensity = 0.0;
		if (intensities_map.find(au_name) != intensities_map.end())
		{
			intensity = intensities_map[au_name];
		}
		else
		{
			// If we do not have an intensity label, trust the occurence one
			intensity = occurences_map[au_name] == 0 ? 0 : 5;
		}

		// Range is now ALWAYS between 0 and 1
		neediness = neediness + ((intensity / 5.0) / 18.0);

		aus[au_name] = std::make_pair(occurence, intensity);
	}

	// INTERRUPT PREWORK
	cv::Matx33f rot = Euler2RotationMatrix(cv::Vec3f(pose[3], pose[4], pose[5]));
	cv::Vec3f rpy = RotationMatrix2Euler(rot);

	double interrupt_raw = pow(rpy[0] - old_pose[0], 2) + pow(rpy[1] - old_pose[1], 2) + pow(rpy[2] - old_pose[2], 2);
	interrupt_raw = (double)1.0/((double)1.0+exp(double(-interrupt_raw)));

	old_pose = rpy;

	// Check if there's currently a reading before doing anything to the values
	bool need_present = neediness > 0 ? .5 : 0;	
	bool interrupt_present = interrupt_raw > 0 ? .5 : 0;

	// HARSH CHECKS ON UPPER BOUNDS
	if (neediness > 1) 		{	neediness = 1;	}
	if (interrupt_raw > 1) 	{	interrupt_raw = 1;	}

	if (neediness < 0) 		{	neediness = 0.0001;	}
	if (interrupt_raw < 0) 	{	interrupt_raw = 0.0001;	}

	neediness = neediness;

	// STORE AND DISPLAY AVERAGE
	// init these with the first value in the array
	float avgNeed = neediness;
	float avgInterrupt = interrupt_raw;
	int averageIndex = window_size - average_size;


	if (personId == 0)
	{
		for (size_t jdx = 0; jdx < window_size - 1; jdx++) {
			needLog_0[jdx] = needLog_0[jdx + 1];
			needLogSmooth_0[jdx] = needLog_0[jdx + 1];
			
			interruptLog_0[jdx] = interruptLog_0[jdx + 1];
			interruptLogSmooth_0[jdx] = interruptLogSmooth_0[jdx + 1];
			
			if (jdx > averageIndex) {
				avgNeed = avgNeed + needLog_0[jdx + 1];
				avgInterrupt = avgInterrupt + interruptLog_0[jdx + 1];

			}

		}

		needLog_0[window_size - 1] = interrupt_raw;
		interruptLog_0[window_size - 1] = neediness;

		avgNeed = avgNeed / average_size;
		avgInterrupt = avgInterrupt / average_size;

		needLogSmooth_0[window_size - 1] = avgNeed;
		interruptLogSmooth_0[window_size - 1] = avgInterrupt;
	} else if (personId == 1)
	{
		for (size_t jdx = 0; jdx < window_size - 1; jdx++) {
			needLog_1[jdx] = needLog_1[jdx + 1];
			needLogSmooth_1[jdx] = needLog_1[jdx + 1];
			
			interruptLog_1[jdx] = interruptLog_1[jdx + 1];
			interruptLogSmooth_1[jdx] = interruptLogSmooth_1[jdx + 1];
			
			if (jdx > averageIndex) {
				avgNeed = avgNeed + needLog_1[jdx + 1];
				avgInterrupt = avgInterrupt + interruptLog_1[jdx + 1];

			}

		}

		needLog_1[window_size - 1] = interrupt_raw;
		interruptLog_1[window_size - 1] = neediness;

		avgNeed = avgNeed / average_size;
		avgInterrupt = avgInterrupt / average_size;

		needLogSmooth_1[window_size - 1] = avgNeed;
		interruptLogSmooth_1[window_size - 1] = avgInterrupt;
	} else if (personId == 2)
	{
		for (size_t jdx = 0; jdx < window_size - 1; jdx++) {
			needLog_2[jdx] = needLog_2[jdx + 1];
			needLogSmooth_2[jdx] = needLog_2[jdx + 1];
			
			interruptLog_2[jdx] = interruptLog_2[jdx + 1];
			interruptLogSmooth_2[jdx] = interruptLogSmooth_2[jdx + 1];
			
			if (jdx > averageIndex) {
				avgNeed = avgNeed + needLog_2[jdx + 1];
				avgInterrupt = avgInterrupt + interruptLog_2[jdx + 1];

			}

		}

		needLog_2[window_size - 1] = interrupt_raw;
		interruptLog_2[window_size - 1] = neediness;

		avgNeed = avgNeed / average_size;
		avgInterrupt = avgInterrupt / average_size;

		needLogSmooth_2[window_size - 1] = avgNeed;
		interruptLogSmooth_2[window_size - 1] = avgInterrupt;
	} else if (personId == 3)
	{
		for (size_t jdx = 0; jdx < window_size - 1; jdx++) {
			needLog_3[jdx] = needLog_3[jdx + 1];
			needLogSmooth_3[jdx] = needLog_3[jdx + 1];
			
			interruptLog_3[jdx] = interruptLog_3[jdx + 1];
			interruptLogSmooth_3[jdx] = interruptLogSmooth_3[jdx + 1];
			
			if (jdx > averageIndex) {
				avgNeed = avgNeed + needLog_3[jdx + 1];
				avgInterrupt = avgInterrupt + interruptLog_3[jdx + 1];

			}

		}

		needLog_3[window_size - 1] = interrupt_raw;
		interruptLog_3[window_size - 1] = neediness;

		avgNeed = avgNeed / average_size;
		avgInterrupt = avgInterrupt / average_size;

		needLogSmooth_3[window_size - 1] = avgNeed;
		interruptLogSmooth_3[window_size - 1] = avgInterrupt;
	} else if (personId == 4)
	{
		for (size_t jdx = 0; jdx < window_size - 1; jdx++) {
			needLog_4[jdx] = needLog_4[jdx + 1];
			needLogSmooth_4[jdx] = needLog_4[jdx + 1];
			
			interruptLog_4[jdx] = interruptLog_4[jdx + 1];
			interruptLogSmooth_4[jdx] = interruptLogSmooth_4[jdx + 1];
			
			if (jdx > averageIndex) {
				avgNeed = avgNeed + needLog_4[jdx + 1];
				avgInterrupt = avgInterrupt + interruptLog_4[jdx + 1];

			}

		}

		needLog_4[window_size - 1] = interrupt_raw;
		interruptLog_4[window_size - 1] = neediness;

		avgNeed = avgNeed / average_size;
		avgInterrupt = avgInterrupt / average_size;

		needLogSmooth_4[window_size - 1] = avgNeed;
		interruptLogSmooth_4[window_size - 1] = avgInterrupt;
	} else if (personId == 5)
	{
		for (size_t jdx = 0; jdx < window_size - 1; jdx++) {
			needLog_5[jdx] = needLog_5[jdx + 1];
			needLogSmooth_5[jdx] = needLog_5[jdx + 1];
			
			interruptLog_5[jdx] = interruptLog_5[jdx + 1];
			interruptLogSmooth_5[jdx] = interruptLogSmooth_5[jdx + 1];
			
			if (jdx > averageIndex) {
				avgNeed = avgNeed + needLog_5[jdx + 1];
				avgInterrupt = avgInterrupt + interruptLog_5[jdx + 1];

			}

		}

		needLog_5[window_size - 1] = interrupt_raw;
		interruptLog_5[window_size - 1] = neediness;

		avgNeed = avgNeed / average_size;
		avgInterrupt = avgInterrupt / average_size;

		needLogSmooth_5[window_size - 1] = avgNeed;
		interruptLogSmooth_5[window_size - 1] = avgInterrupt;

	} else if (personId == 6)
	{
		for (size_t jdx = 0; jdx < window_size - 1; jdx++) {
			needLog_6[jdx] = needLog_6[jdx + 1];
			needLogSmooth_6[jdx] = needLog_6[jdx + 1];
			
			interruptLog_6[jdx] = interruptLog_6[jdx + 1];
			interruptLogSmooth_6[jdx] = interruptLogSmooth_6[jdx + 1];
			
			if (jdx > averageIndex) {
				avgNeed = avgNeed + needLog_6[jdx + 1];
				avgInterrupt = avgInterrupt + interruptLog_6[jdx + 1];

			}

		}

		needLog_6[window_size - 1] = interrupt_raw;
		interruptLog_6[window_size - 1] = neediness;

		avgNeed = avgNeed / average_size;
		avgInterrupt = avgInterrupt / average_size;

		needLogSmooth_6[window_size - 1] = avgNeed;
		interruptLogSmooth_6[window_size - 1] = avgInterrupt;
	}


	// If you want to turn off value smoothing/averaging,in the display 
	// you can do that here
	if (false) {
		neediness = avgNeed;
		interrupt_raw = avgInterrupt;
	}
	// Otherwise just pick which need graph to display

	// MODULATE for display if we so choose
	double need_intensity = neediness;
	double interrupt_intensity = interrupt_raw;

	classifications["neediness"] = std::make_pair(neediness > 0 ? 1 : 0, neediness);

	int idx = (numberOfSlots * personId);
	std::string need_name = "Neediness " + std::to_string(personId + 1);
	
	cv::Scalar need_color = color_level_0;
	if (need_intensity > .8) {need_color = color_level_4;}
	else if (need_intensity > .6) {need_color = color_level_3;}
	else if (need_intensity > .4) {need_color = color_level_2;}
	else if (need_intensity > .2) {need_color = color_level_1;}
	else if (need_intensity > .0) {need_color = color_level_0;}

	// ADD THESE METRICS TO THE GUI
	int offset = MARGIN_Y + idx * (AU_TRACKBAR_HEIGHT + 10);
	std::ostringstream au_i_need;
	au_i_need << std::setprecision(2) << std::setw(4) << std::fixed << need_intensity;
	cv::putText(classifier_image, need_name, cv::Point(10, offset + 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, person_color, 1, cv::LINE_AA);
	//cv::putText(classifier_image, need_name, cv::Point(55, offset + 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0, 0, 0), 1, cv::LINE_AA);

	if (need_present)
	{
		cv::putText(classifier_image, au_i_need.str(), cv::Point(160, offset + 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, person_color, 1, cv::LINE_AA);
		cv::rectangle(classifier_image, cv::Point(MARGIN_X, offset),
			cv::Point((int)(MARGIN_X + AU_TRACKBAR_LENGTH * need_intensity), offset + AU_TRACKBAR_HEIGHT),
			need_color,
			cv::FILLED);
	}
	else
	{
		cv::putText(classifier_image, "0.00", cv::Point(160, offset + 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0, 0, 0), 1, cv::LINE_AA);
	}

	// INTERRUPTION POSTING

	int idx2 = (numberOfSlots * personId) + 1;
	std::string interrupt_name = "Interruptibility " + std::to_string(personId + 1);

	// TODO: Decide if want different colors for interrupt colors
	cv::Scalar interrupt_color = color_level_0;
	if (interrupt_intensity > .8) {interrupt_color = color_level_0;}
	else if (interrupt_intensity > .6) {interrupt_color = color_level_1;}
	else if (interrupt_intensity > .4) {interrupt_color = color_level_2;}
	else if (interrupt_intensity > .2) {interrupt_color = color_level_3;}
	else if (interrupt_intensity > .0) {interrupt_color = color_level_4;}

	// ADD THESE METRICS TO THE GUI
	int offset_interrupt = MARGIN_Y + idx2 * (AU_TRACKBAR_HEIGHT + 10);
	std::ostringstream au_i_interrupt;
	au_i_interrupt << std::setprecision(2) << std::setw(4) << std::fixed << interrupt_intensity;
	cv::putText(classifier_image, interrupt_name, cv::Point(10, offset_interrupt + 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, person_color, 1, cv::LINE_AA);
	//cv::putText(classifier_image, need_name, cv::Point(55, offset + 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0, 0, 0), 1, cv::LINE_AA);

	if (interrupt_present)
	{
		cv::putText(classifier_image, au_i_interrupt.str(), cv::Point(160, offset_interrupt + 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, person_color, 1, cv::LINE_AA);
		cv::rectangle(classifier_image, cv::Point(MARGIN_X, offset_interrupt),
			cv::Point((int)(MARGIN_X + AU_TRACKBAR_LENGTH * interrupt_intensity), offset_interrupt + AU_TRACKBAR_HEIGHT),
			interrupt_color,
			cv::FILLED);
	}
	else
	{
		cv::putText(classifier_image, "0.00", cv::Point(160, offset_interrupt + 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0, 0, 0), 1, cv::LINE_AA);
	}


	// Log for future use and potential graph
	//needLog[personId].push_back(neediness);
	//interruptLog[personId].push_back(interrupt_raw);

	// cv::Mat tempLog;
	// VectorToMat(needLog[personId],  tempLog);

	// cv::Ptr<cv::plot::Plot2d> plot = cv::plot::createPlot2d(needLog);
	// plot->render(graph_image);
	// cv::imshow( "Need Log", graph_image );

}

//Create separate window, and display a graph of all the currently visible action units
void GastroViz::SetObservationActionUnits(const std::vector<std::pair<std::string, double> >& au_intensities,
	const std::vector<std::pair<std::string, double> >& au_occurences)
{
	// For each of the noticed AUs
	if (au_intensities.size() > 0 || au_occurences.size() > 0)
	{

		std::set<std::string> au_names;
		std::map<std::string, bool> occurences_map;
		std::map<std::string, double> intensities_map;

		for (size_t idx = 0; idx < au_intensities.size(); idx++)
		{
			au_names.insert(au_intensities[idx].first);
			intensities_map[au_intensities[idx].first] = au_intensities[idx].second;
		}

		for (size_t idx = 0; idx < au_occurences.size(); idx++)
		{
			au_names.insert(au_occurences[idx].first);
			occurences_map[au_occurences[idx].first] = au_occurences[idx].second > 0;
		}

		const int AU_TRACKBAR_LENGTH = 400;
		const int AU_TRACKBAR_HEIGHT = 10;

		const int MARGIN_X = 185;
		const int MARGIN_Y = 10;

		const int nb_aus = (int) au_names.size();

		// Do not reinitialize
		if (action_units_image.empty())
		{
			action_units_image = cv::Mat(nb_aus * (AU_TRACKBAR_HEIGHT + 10) + MARGIN_Y * 2, AU_TRACKBAR_LENGTH + MARGIN_X, CV_8UC3, cv::Scalar(255, 255, 255));
		}
		else
		{
			action_units_image.setTo(255);
		}

		std::map<std::string, std::pair<bool, double>> aus;

		// first, prepare a mapping "AU name" -> { present, intensity }
		for (auto au_name : au_names)
		{
			// Insert the intensity and AU presense (as these do not always overlap check if they exist first)
			bool occurence = false;
			if (occurences_map.find(au_name) != occurences_map.end())
			{
				occurence = occurences_map[au_name] != 0;
			}
			else
			{
				// If we do not have an occurence label, trust the intensity one
				occurence = intensities_map[au_name] > 1;
			}
			double intensity = 0.0;
			if (intensities_map.find(au_name) != intensities_map.end())
			{
				intensity = intensities_map[au_name];
			}
			else
			{
				// If we do not have an intensity label, trust the occurence one
				intensity = occurences_map[au_name] == 0 ? 0 : 5;
			}

			aus[au_name] = std::make_pair(occurence, intensity);
		}

		// then, build the graph
		unsigned int idx = 0;
		for (auto& au : aus)
		{
			std::string name = au.first;
			bool present = au.second.first;
			double intensity = au.second.second;

			int offset = MARGIN_Y + idx * (AU_TRACKBAR_HEIGHT + 10);
			std::ostringstream au_i;
			au_i << std::setprecision(2) << std::setw(4) << std::fixed << intensity;
			cv::putText(action_units_image, name, cv::Point(10, offset + 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(present ? 0 : 200, 0, 0), 1, cv::LINE_AA);
			cv::putText(action_units_image, AUS_DESCRIPTION.at(name), cv::Point(55, offset + 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0, 0, 0), 1, cv::LINE_AA);

			if (present)
			{
				cv::putText(action_units_image, au_i.str(), cv::Point(160, offset + 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0, 100, 0), 1, cv::LINE_AA);
				cv::rectangle(action_units_image, cv::Point(MARGIN_X, offset),
					cv::Point((int)(MARGIN_X + AU_TRACKBAR_LENGTH * intensity / 5.0), offset + AU_TRACKBAR_HEIGHT),
					cv::Scalar(128, 128, 128),
					cv::FILLED);
			}
			else
			{
				cv::putText(action_units_image, "0.00", cv::Point(160, offset + 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0, 0, 0), 1, cv::LINE_AA);
			}
			idx++;
		}
	}
}


// Eye gaze infomration drawing, first of eye landmarks then of gaze
void GastroViz::SetObservationGaze(const cv::Point3f& gaze_direction0, const cv::Point3f& gaze_direction1, const std::vector<cv::Point2f>& eye_landmarks2d, const std::vector<cv::Point3f>& eye_landmarks3d, double confidence)
{
	if(confidence > visualisation_boundary)
	{
		if (eye_landmarks2d.size() > 0)
		{
			// First draw the eye region landmarks
			for (size_t i = 0; i < eye_landmarks2d.size(); ++i)
			{
				cv::Point featurePoint(cvRound(eye_landmarks2d[i].x * (double)draw_multiplier), cvRound(eye_landmarks2d[i].y * (double)draw_multiplier));

				// A rough heuristic for drawn point size
				int thickness = 1;
				int thickness_2 = 1;

				size_t next_point = i + 1;
				if (i == 7)
					next_point = 0;
				if (i == 19)
					next_point = 8;
				if (i == 27)
					next_point = 20;

				if (i == 7 + 28)
					next_point = 0 + 28;
				if (i == 19 + 28)
					next_point = 8 + 28;
				if (i == 27 + 28)
					next_point = 20 + 28;

				cv::Point nextFeaturePoint(cvRound(eye_landmarks2d[next_point].x * (double)draw_multiplier), cvRound(eye_landmarks2d[next_point].y * (double)draw_multiplier));
				if ((i < 28 && (i < 8 || i > 19)) || (i >= 28 && (i < 8 + 28 || i > 19 + 28)))
					cv::line(captured_image, featurePoint, nextFeaturePoint, cv::Scalar(255, 0, 0), thickness_2, cv::LINE_AA, draw_shiftbits);
				else
					cv::line(captured_image, featurePoint, nextFeaturePoint, cv::Scalar(0, 0, 255), thickness_2, cv::LINE_AA, draw_shiftbits);

			}

			// Now draw the gaze lines themselves
			cv::Mat cameraMat = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 0);

			// Grabbing the pupil location, to draw eye gaze need to know where the pupil is
			cv::Point3f pupil_left(0, 0, 0);
			cv::Point3f pupil_right(0, 0, 0);
			for (size_t i = 0; i < 8; ++i)
			{
				pupil_left = pupil_left + eye_landmarks3d[i];
				pupil_right = pupil_right + eye_landmarks3d[i + eye_landmarks3d.size()/2];
			}
			pupil_left = pupil_left / 8;
			pupil_right = pupil_right / 8;

			std::vector<cv::Point3f> points_left;
			points_left.push_back(cv::Point3f(pupil_left));
			points_left.push_back(cv::Point3f(pupil_left) + cv::Point3f(gaze_direction0)*50.0);

			std::vector<cv::Point3f> points_right;
			points_right.push_back(cv::Point3f(pupil_right));
			points_right.push_back(cv::Point3f(pupil_right) + cv::Point3f(gaze_direction1)*50.0);

			cv::Mat_<float> proj_points;
			cv::Mat_<float> mesh_0 = (cv::Mat_<float>(2, 3) << points_left[0].x, points_left[0].y, points_left[0].z, points_left[1].x, points_left[1].y, points_left[1].z);
			Project(proj_points, mesh_0, fx, fy, cx, cy);
			cv::line(captured_image, cv::Point(cvRound(proj_points.at<float>(0, 0) * (float)draw_multiplier), cvRound(proj_points.at<float>(0, 1) * (float)draw_multiplier)),
				cv::Point(cvRound(proj_points.at<float>(1, 0) * (float)draw_multiplier), cvRound(proj_points.at<float>(1, 1) * (float)draw_multiplier)), cv::Scalar(110, 220, 0), 2, cv::LINE_AA, draw_shiftbits);

			cv::Mat_<float> mesh_1 = (cv::Mat_<float>(2, 3) << points_right[0].x, points_right[0].y, points_right[0].z, points_right[1].x, points_right[1].y, points_right[1].z);
			Project(proj_points, mesh_1, fx, fy, cx, cy);
			cv::line(captured_image, cv::Point(cvRound(proj_points.at<float>(0, 0) * (float)draw_multiplier), cvRound(proj_points.at<float>(0, 1) * (float)draw_multiplier)),
				cv::Point(cvRound(proj_points.at<float>(1, 0) * (float)draw_multiplier), cvRound(proj_points.at<float>(1, 1) * (float)draw_multiplier)), cv::Scalar(110, 220, 0), 2, cv::LINE_AA, draw_shiftbits);

		}
	}
}

// Label the image with the framerate
void GastroViz::SetFps(double fps)
{
	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps);
	std::string fpsSt("FPS:");
	fpsSt += fpsC;
	cv::putText(captured_image, fpsSt, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, cv::LINE_AA);
}


// The code that actually displays each of the elements in their windows
// Each imshow launches its own new window
char GastroViz::ShowObservation()
{
	bool ovservation_shown = false;

	if (!graph_image.empty()) {
		cv::imshow("Need Over Time Graph", graph_image);
	}

	if (!classifier_image.empty()) {
		cv::imshow("Interaction Classifications", classifier_image);
	}

	if (vis_align && !aligned_face_image.empty())
	{
		cv::imshow("sim_warp", aligned_face_image);
		ovservation_shown = true;
	}
	if (vis_hog && !hog_image.empty())
	{
		cv::imshow("HOG", hog_image);
		ovservation_shown = true;
	}
	if (vis_aus && !action_units_image.empty())
	{
		cv::imshow("In-Depth Action Units", action_units_image);
		ovservation_shown = true;
	}
	if (vis_track)
	{
		cv::imshow("Tracking Result", captured_image);
		ovservation_shown = true;
	}
	
	// Only perform waitKey if something was shown
	char result = '\0';
	if (ovservation_shown)
	{
		result = cv::waitKey(1);
	}
	return result;

}

// Get the normal, unaltered image for changes	
cv::Mat GastroViz::GetVisImage()
{
	return captured_image;
}

cv::Mat GastroViz::GetHOGVis()
{
	return hog_image;
}


void GastroViz::VectorToMat(const std::vector<float>& in,  cv::Mat& out)    
{
	std::vector<float>::const_iterator it = in.begin();
	cv::MatIterator_<uchar> jt, end;
	jt = out.begin<uchar>();
	for (; it != in.end(); ++it) { *jt++ = (uchar)(*it * 255); } 
}

// A sample function for graphing a histogram on an input image
// Graphs a histogram of the colors in the original histogram
void GastroViz::showHistogram(cv::Mat src, cv::Mat &hist_image)
{ // based on http://docs.opencv.org/2.4.4/modules/imgproc/doc/histograms.html?highlight=histogram#calchist

   int sbins = 256;
   int histSize[] = {sbins};

   float sranges[] = { 0, 256 };
   const float* ranges[] = { sranges };
   cv::MatND hist;
   int channels[] = {0};

   cv::calcHist( &src, 1, channels, cv::Mat(), // do not use mask
       hist, 1, histSize, ranges,
       true, // the histogram is uniform
       false );

   double maxVal=0;
   minMaxLoc(hist, 0, &maxVal, 0, 0);

   int xscale = 10;
   int yscale = 10;
   //hist_image.create(
   hist_image = cv::Mat::zeros(256, sbins*xscale, CV_8UC3);

   for( int s = 0; s < sbins; s++ )
   {
       float binVal = hist.at<float>(s, 0);
       int intensity = cvRound(binVal*255/maxVal);
       rectangle( hist_image, cv::Point(s*xscale, 0),
           cv::Point( (s+1)*xscale - 1, intensity),
           cv::Scalar::all(255),
           CV_FILLED );
   }
}

void GastroViz::ShowNeedGraph(int personId)
{ // based on http://docs.opencv.org/2.4.4/modules/imgproc/doc/histograms.html?highlight=histogram#calchist

	int yscale = 200;

	if (graph_image.empty() || personId == 0)
	{
		//To do, handle if multiple people
		graph_image = cv::Mat(MARGIN_Y * 2 + yscale, 2 * MARGIN_X + window_size*AU_TRACKBAR_HEIGHT, CV_8UC3, cv::Scalar(50, 50, 50));
	}

	// Get the correct color for the person
	cv::Scalar person_color;
	float *needLog;

	if (personId == 0) {
		person_color = person_color_0;
		needLog = needLogSmooth_0;		
	} else if (personId == 1) {
		person_color = person_color_1;
		needLog = needLogSmooth_1;		
	} else if (personId == 2) {
		person_color = person_color_2;
		needLog = needLogSmooth_2;		
	} else if (personId == 3) {
		person_color = person_color_3;
		needLog = needLogSmooth_3;		
	} else if (personId == 4) {
		person_color = person_color_4;
		needLog = needLogSmooth_4;	
	} else {
		person_color = person_color_5;
		needLog = needLogSmooth_5;
	} 

	int sbins = window_size;

	int indexA, indexB;
	float dataA, dataB;
	cv::Point ptA, ptB;

   for( int s = 0; s < sbins - 1; s++ )
   {
	    indexA = s;
	    indexB = s + 1;

       	dataA = needLog[indexA];
	   	dataB = needLog[indexB];

		ptA = cv::Point(indexA * (AU_TRACKBAR_HEIGHT * 2), yscale - (dataA * yscale) + MARGIN_Y);
		ptB = cv::Point(indexB * (AU_TRACKBAR_HEIGHT * 2), yscale - (dataB * yscale) + MARGIN_Y);

       	int intensity = 1;
    	line(graph_image, ptA, ptB, person_color, 1, 8, 0);
   }
}
