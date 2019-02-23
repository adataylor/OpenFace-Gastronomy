# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
add_executable(DemoSingle DemoSingle.cpp)
target_link_libraries(DemoSingle LandmarkDetector)
target_link_libraries(DemoSingle FaceAnalyser)
target_link_libraries(DemoSingle GazeAnalyser)
target_link_libraries(DemoSingle Utilities)

install (TARGETS DemoSingle DESTINATION bin)