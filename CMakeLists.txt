cmake_minimum_required(VERSION 3.14)

project(ongp)

list(APPEND CMAKE_PREFIX_PATH "/nerf/openNGP_3rdparty/libtorch-cpu")
list(APPEND CMAKE_PREFIX_PATH "/nerf/openNGP_3rdparty/gtest")
message(${CMAKE_PREFIX_PATH})

# libtorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Include the subdirectories
add_subdirectory(ongp)
add_subdirectory(test/cpp)