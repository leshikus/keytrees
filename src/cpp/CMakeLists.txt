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

# exact_cost: exact recurrences for search-tree variants and brute-force checks
# rta:        C++23 port of the 1998 GENDATA | RTA experiment pipeline
# ecc:        prototype for identification keys that tolerate t test errors
add_executable(exact_cost exact_cost.cpp)
add_executable(rta rta.cpp)
add_executable(ecc ecc.cpp)
