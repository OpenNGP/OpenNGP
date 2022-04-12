
set(MAIN app/main.cc)
file(GLOB_RECURSE  ONGP_SRCS CONFIGURE_DEPENDS "*.h" "*.hpp" "*.cc")

add_executable(ongp ${MAIN} ${ONGP_SRCS})

target_link_libraries(ongp "${TORCH_LIBRARIES}")
set_property(TARGET ongp PROPERTY CXX_STANDARD 14)