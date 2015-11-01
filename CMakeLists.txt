cmake_minimum_required(VERSION 2.8)
project(OnlineX)

include_directories("util")
include_directories("lib")
add_subdirectory("util")
add_subdirectory("lib")

IF(CMAKE_COMPILER_IS_GNUCXX)
    message("Found g++ version " ${CMAKE_CXX_COMPILER_VERSION})
    ADD_DEFINITIONS("-std=c++11" "-g" "-O3")
ELSE("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    message("Found clang++ version" ${CMAKE_CXX_COMPILER_VERSION})
    ADD_DEFINITIONS("-std=c++11" "-g" "-O3" "-Wno-deprecated-register")
ELSE()
MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()

ADD_LIBRARY(OnlineX SHARED
#    src/utils/Sample.cpp
#    src/utils/cokus.cpp
#    src/utils/objcokus.cpp
    lib/OnlineX.cpp
#    src/feed_one.cpp
)

add_executable(cppWrapper
    cppWrapper.cpp
)

target_link_libraries(cppWrapper
    OnlineX
)
