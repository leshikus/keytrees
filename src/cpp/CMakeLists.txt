cmake_minimum_required(VERSION 3.28)
project(codes_engines LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
if(WIN32)
  add_link_options(-static)  # the binaries then run without the toolchain on PATH
endif()

# One executable per source file in this directory, named after the file:
# exact_cost (exact recurrences for search-tree variants and brute-force checks),
# rta (C++23 port of the 1998 GENDATA | RTA pipeline), ecc (keys that tolerate
# t test errors), and the engines added for the articles in articles/.
file(GLOB engine_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(source IN LISTS engine_sources)
  get_filename_component(engine ${source} NAME_WE)
  add_executable(${engine} ${source})
endforeach()
