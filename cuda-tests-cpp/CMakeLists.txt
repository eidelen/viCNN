cmake_minimum_required(VERSION 3.18)
project(cuda_minimal LANGUAGES CUDA)

# Set the standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add the executable
add_executable(cuda_minimal src/main.cu)

# Optional: Only compile for your architecture
# set_target_properties(cuda_minimal PROPERTIES CUDA_ARCHITECTURES 86)
