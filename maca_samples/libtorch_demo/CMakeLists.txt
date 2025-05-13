cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch_demo)


# !!! please set this option when using MACA platform
option(USE_MACA "Use MACA" ON)


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(${PROJECT_NAME} libtorch_demo.cpp)
# !!! please set this option when using MACA platform
target_compile_definitions(${PROJECT_NAME} PRIVATE USE_MACA)


target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

