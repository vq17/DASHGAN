cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_library(DASHGAN SHARED DASHGAN-dylib.cpp)
target_link_libraries(DASHGAN "${TORCH_LIBRARIES}")
set_property(TARGET DASHGAN PROPERTY CXX_STANDARD 14)
