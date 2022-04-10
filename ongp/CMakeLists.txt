
add_executable(ongp app/main.cpp)

target_link_libraries(ongp "${TORCH_LIBRARIES}")
set_property(TARGET ongp PROPERTY CXX_STANDARD 14)