cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)
set(CMAKE_BUILD_TYPE "Release")

set(ADD_TEST ON)
find_package(Torch REQUIRED)

if(${ADD_TEST})
    set(GTEST_DIR ${CMAKE_SOURCE_DIR}/../../third_party/googletest/googletest/include)
    set(GMOCK_DIR ${CMAKE_SOURCE_DIR}/../../third_party/googletest/googlemock/include)
    include_directories(${GTEST_DIR})
    include_directories(${GMOCK_DIR})

    set(LIBS_DIR ${CMAKE_SOURCE_DIR}/../../build/lib)

    find_library(GTEST_LIBRARY gtest PATHS "${LIBS_DIR}")
    find_library(GTEST_MAIN_LIBRARY gtest_main PATHS "${LIBS_DIR}")
    list(APPEND TORCH_LIBRARIES ${GTEST_LIBRARY})
    list(APPEND TORCH_LIBRARIES ${GTEST_MAIN_LIBRARY})
    find_library(GMOCK_LIBRARY gmock PATHS "${LIBS_DIR}")
    find_library(GMOCK_MAIN_LIBRARY gmock_main PATHS "${LIBS_DIR}")
    list(APPEND TORCH_LIBRARIES ${GMOCK_LIBRARY})
    list(APPEND TORCH_LIBRARIES ${GMOCK_MAIN_LIBRARY})

    add_subdirectory(${PROJECT_SOURCE_DIR}/test_api)
endif()


