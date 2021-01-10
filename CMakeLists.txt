cmake_minimum_required(VERSION 3.16)
project(kay)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/")
find_package(Python 3.8 COMPONENTS Development Interpreter REQUIRED)
add_subdirectory(pybind11)

find_package(Embree REQUIRED)

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${EMBREE_INCLUDE_PATH})
include_directories(pybind11/include)


add_compile_options(-Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)

set(SRCS src/kay.cpp
        src/render.cpp
        src/render.h
        src/ptr.h)

pybind11_add_module(kay ${SRCS})

target_link_libraries(kay PRIVATE ${EMBREE_LIBRARY})


