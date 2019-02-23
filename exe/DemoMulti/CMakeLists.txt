# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
add_executable(DemoMulti DemoMulti.cpp)
target_link_libraries(DemoMulti LandmarkDetector)
target_link_libraries(DemoMulti FaceAnalyser)
target_link_libraries(DemoMulti GazeAnalyser)
target_link_libraries(DemoMulti Utilities)

install (TARGETS DemoMulti DESTINATION bin)