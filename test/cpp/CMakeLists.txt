
project(unittest)
find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIR})

add_executable(unittest test_data_loader.cc 
                        test_camera.cc
                        test_tensor.cc)
target_link_libraries(unittest ${TORCH_LIBRARIES} ${GTEST_BOTH_LIBRARIES}) 
enable_testing()