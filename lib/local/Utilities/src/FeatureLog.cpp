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

#include "FeatureLog.h"
#include "ImageManipulationHelpers.h"
#include <iostream>
#include <random>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>

// OpenCV includes
#include <opencv2/imgproc.hpp>

using namespace Utilities;

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

bool FeatureLog::Setup(std::vector<std::string>& arguments)
{
	
	return false;
}

// Input head factors, AUs
bool FeatureLog::Log(const std::vector<std::pair<std::string, double> >& au_intensities,
	const std::vector<std::pair<std::string, double> >& au_occurences, const cv::Vec6f& pose, double confidence)
{
	// Log head rotations over time
	// pose_Rx, pose_Ry, pose_Rz


	// Log interesting action units over time

	float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);	
	needLog.push_back(r);
}

std::vector<float> FeatureLog::getNeedLog() 
{

	return needLog;

}
