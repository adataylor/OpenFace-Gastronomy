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

// For drawing on images
#include <opencv2/imgproc.hpp>

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

// Set up the GastroViz instance
GastroViz::GastroViz(std::vector<std::string> arguments)
{
	// By default not visualizing anything
	// FALSE, by default we are visualizing AUs
	// TODO check if also visualizing some other things	
	this->vis_track = false;
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

	top_view_image = cv::Mat();

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
void GastroViz::SetObservationPose(const cv::Vec6f& pose, double confidence)
{

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
		DrawBox(captured_image, pose, cv::Scalar(vis_certainty*255.0, 0, (1 - vis_certainty) * 255), thickness, fx, fy, cx, cy);
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
cv::Scalar pointColor1 = cv::Scalar(0, 255, 255);
cv::Scalar pointColor2 = cv::Scalar(255, 0, 255);
cv::Scalar pointColor3 = cv::Scalar(0, 255, 0);
cv::Scalar pointColor4 = cv::Scalar(255, 0, 0);

cv::Scalar indicatorColor = pointColor3;

//Draw an indicator circle
// TODO turn into Sims-style indicator?
// circle(captured_image, humanPoint, radiusPoint, pointColor, thickness=1, cv::LINE_AA, shift=0);
	

}

void GastroViz::SetTopView(const cv::Vec6f& pose, double confidence, const std::vector<std::pair<std::string, double> >& au_intensities,
	const std::vector<std::pair<std::string, double> >& au_occurences)
{
	if (top_view_image.empty())
	{
		top_view_image = cv::Mat(300, 300, cv::Scalar(255, 255, 255));
	}

	// DISTANCES are in mm
	// Use 1/10 of a meter incremements?

	cv::Scalar need1 = cv::Scalar(255, 0, 255);
	cv::Scalar need2 = cv::Scalar(0, 0, 255);
	cv::Scalar need3 = cv::Scalar(255, 0, 0);


	float pX = 100.0 * pose[0] / 1000.0;
	float pY = 100.0 * pose[1] / 1000.0;
	float pZ = 100.0 * pose[2] / 1000.0;

	cv::Point humanPoint = cv::Point((int)pX, (int)pY);

	cv::circle(top_view_image,
        humanPoint,
        5,
        need2,
        5,
        cv::LINE_AA);

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

	if (vis_align && !aligned_face_image.empty())
	{
		cv::imshow("sim_warp", aligned_face_image);
		ovservation_shown = true;
	}
	if (vis_hog && !hog_image.empty())
	{
		cv::imshow("hog", hog_image);
		ovservation_shown = true;
	}
	if (vis_aus && !action_units_image.empty())
	{
		cv::imshow("action units", action_units_image);
		ovservation_shown = true;
	}
	if (vis_track)
	{
		cv::imshow("tracking result", captured_image);
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


