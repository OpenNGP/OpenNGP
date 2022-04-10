cmake_minimum_required(VERSION 3.14)

project(ongp)

#set(CMAKE_PREFIX_PATH "/nerf/openNGP_3rdParty/libtorch")
# libtorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# Include the subdirectories
add_subdirectory(ongp)