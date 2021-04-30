cmake_minimum_required(VERSION 3.19)
project(cuda_test)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)

add_executable(cuda_test main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(cuda_test ${OpenCV_LIBS})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIB_DIR})