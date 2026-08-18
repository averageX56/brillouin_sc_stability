cmake_minimum_required(VERSION 3.24)
project(brillouin_sde_cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(sde_solver_cuda sde_solver_cuda.cu)
target_compile_features(sde_solver_cuda PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(sde_solver_cuda PRIVATE CUDA::cudart CUDA::curand)
set_target_properties(sde_solver_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES native)
target_compile_options(sde_solver_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
