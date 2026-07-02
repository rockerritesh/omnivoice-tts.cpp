cmake_minimum_required(VERSION 3.16)
project(omnivoice_cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CPU-only first, for bit-approximate parity against the numpy/Rust reference.
# Pass -DOMNI_CUDA=ON to build the GPU engine (tts_cuda) with the ggml CUDA backend.
option(OMNI_CUDA "Build tts_cuda with ggml CUDA backend" OFF)
option(OMNI_METAL "Build tts_cuda with ggml Metal (Apple GPU) backend" OFF)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(OMNI_CUDA)
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(OMNI_METAL)
  set(GGML_METAL ON CACHE BOOL "" FORCE)
else()
  set(GGML_METAL OFF CACHE BOOL "" FORCE)  # CPU-default keeps reference parity
endif()

add_subdirectory(vendor/ggml)

add_executable(test_qwen3 src/test_qwen3.cpp)
target_link_libraries(test_qwen3 PRIVATE ggml)
target_include_directories(test_qwen3 PRIVATE vendor/ggml/include)

add_executable(test_codec src/test_codec.cpp)
target_link_libraries(test_codec PRIVATE ggml)
target_include_directories(test_codec PRIVATE vendor/ggml/include)

add_executable(stage0 src/stage0.cpp)
target_link_libraries(stage0 PRIVATE ggml)
target_include_directories(stage0 PRIVATE vendor/ggml/include)

add_executable(test_tokenizer src/test_tokenizer.cpp)

add_executable(tts src/tts.cpp)
target_link_libraries(tts PRIVATE ggml)
target_include_directories(tts PRIVATE vendor/ggml/include)

# GPU-capable engine (backbone via ggml-backend; CUDA when OMNI_CUDA=ON, else CPU backend)
add_executable(tts_cuda src/tts_cuda.cpp)
target_link_libraries(tts_cuda PRIVATE ggml)
target_include_directories(tts_cuda PRIVATE vendor/ggml/include)
if(OMNI_CUDA)
  target_compile_definitions(tts_cuda PRIVATE OMNI_USE_CUDA)
endif()
if(OMNI_METAL)
  target_compile_definitions(tts_cuda PRIVATE OMNI_USE_METAL)
endif()
