cmake_minimum_required(VERSION 3.22)
project(volumetric LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED PATHS /usr/lib/cmake/glm)

add_executable(volumetric
    src/main.cpp
    src/renderer.cu
)

target_link_libraries(volumetric
    CUDA::cudart
    GLEW::GLEW
    glm::glm
)

target_include_directories(volumetric PRIVATE
    /usr/include
    third_party
)

set_target_properties(volumetric PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 80
)