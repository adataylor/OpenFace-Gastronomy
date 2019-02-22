///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis all rights reserved.
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

#ifndef FEATURE_LOG_H
#define FEATURE_LOG_H

// System includes
#include <fstream>
#include <sstream>
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace Utilities
{

	//===========================================================================
	/**
	A class for capturing sequences from video, webcam, and image directories
	*/
	class FeatureLog {

	public:

		// Default constructor
		FeatureLog() {};

		// Opening based on command line arguments
	
		bool Log(const std::vector<std::pair<std::string, double> >& au_intensities,
	const std::vector<std::pair<std::string, double> >& au_occurences, const cv::Vec6f& pose, double confidence);

		std::vector<float> getNeedLog();
		std::vector<float> getInterruptLog();


	private:

		// // Blocking copy and move, as it doesn't make sense to have several readers pointed at the same source
		// ImageCapture & operator= (const ImageCapture& other);
		// ImageCapture & operator= (const ImageCapture&& other);
		// ImageCapture(const ImageCapture&& other);
		// ImageCapture(const ImageCapture& other);

		// Need log
		std::vector<float> needLog;
		std::vector<float> interruptLog;

		std::vector<cv::Vec6f& pose> poseLog;
		
		//TODO add log of AUs

		std::vector<float> handYLog;
		std::vector<float> handXLog;


		// Interrupt log


	};
}
#endif // IMAGE_CAPTURE_H