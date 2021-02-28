cmake_minimum_required(VERSION 3.16)
project(yolo3_mobilnet)

#ncnn
include_directories(include)
link_directories(lib)

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND OR OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++11 " )

endif ()


set(CMAKE_CXX_STANDARD 14)

add_executable(yolo3_mobilnet main.cpp)
target_link_libraries(yolo3_mobilnet ncnn ${OpenCV_LIBS})