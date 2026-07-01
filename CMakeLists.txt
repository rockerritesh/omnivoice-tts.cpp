cmake_minimum_required(VERSION 3.16)
project(omnivoice_cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CPU-only first, for bit-approximate parity against the numpy/Rust reference.
# Metal can be enabled later once parity is established.
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(vendor/ggml)

add_executable(test_qwen3 src/test_qwen3.cpp)
target_link_libraries(test_qwen3 PRIVATE ggml)
target_include_directories(test_qwen3 PRIVATE vendor/ggml/include)

add_executable(test_codec src/test_codec.cpp)
target_link_libraries(test_codec PRIVATE ggml)
target_include_directories(test_codec PRIVATE vendor/ggml/include)
