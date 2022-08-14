cmake_minimum_required(VERSION 3.14)

project(ongp)

#list(APPEND CMAKE_PREFIX_PATH "/nerf/openNGP_3rdparty/libtorch-cpu")
list(APPEND CMAKE_PREFIX_PATH "/nerf/openNGP_3rdparty/libtorch")
list(APPEND CMAKE_PREFIX_PATH "/nerf/openNGP_3rdparty/gtest")
message(${CMAKE_PREFIX_PATH})


add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# libtorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ongp/external)

# Include the subdirectories
add_subdirectory(ongp)
add_subdirectory(test/cpp)