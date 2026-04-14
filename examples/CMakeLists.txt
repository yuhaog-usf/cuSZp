# Find CUDA package
find_package(CUDAToolkit REQUIRED)

# Define the install directory for binaries
set(install_dir ${PROJECT_BINARY_DIR}/examples/bin)
set(cuSZp "cuSZp")
set(cuSZp_test_f32_1D "cuSZp_test_f32_1D")
set(cuSZp_test_f64_1D "cuSZp_test_f64_1D")
set(cuSZp_test_f32_2D "cuSZp_test_f32_2D")
set(cuSZp_test_f64_2D "cuSZp_test_f64_2D")
set(cuSZp_test_f32_3D "cuSZp_test_f32_3D")
set(cuSZp_test_f64_3D "cuSZp_test_f64_3D")
set(cuSZp_bench_1D_f32_plain "cuSZp_bench_1D_f32_plain")
set(cuSZp_profile_compress "cuSZp_profile_compress")
set(cuSZp_profile_decompress "cuSZp_profile_decompress")

# Include the main include directory
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${CUDAToolkit_INCLUDE_DIRS})

# Compile executable binaries
add_executable(${cuSZp} cuSZp.cpp)
add_executable(${cuSZp_test_f32_1D} cuSZp_test_f32_1D.cpp)
add_executable(${cuSZp_test_f64_1D} cuSZp_test_f64_1D.cpp)
add_executable(${cuSZp_test_f32_2D} cuSZp_test_f32_2D.cpp)
add_executable(${cuSZp_test_f64_2D} cuSZp_test_f64_2D.cpp)
add_executable(${cuSZp_test_f32_3D} cuSZp_test_f32_3D.cpp)
add_executable(${cuSZp_test_f64_3D} cuSZp_test_f64_3D.cpp)
add_executable(${cuSZp_bench_1D_f32_plain} cuSZp_bench_1D_f32_plain.cpp)
add_executable(${cuSZp_profile_compress} cuSZp_profile_compress.cpp)
add_executable(${cuSZp_profile_decompress} cuSZp_profile_decompress.cpp)

# Link the existing cuSZp_static library
target_link_libraries(${cuSZp} PRIVATE cuSZp_static)
target_link_libraries(${cuSZp_test_f32_1D} PRIVATE cuSZp_static)
target_link_libraries(${cuSZp_test_f64_1D} PRIVATE cuSZp_static)
target_link_libraries(${cuSZp_test_f32_2D} PRIVATE cuSZp_static)
target_link_libraries(${cuSZp_test_f64_2D} PRIVATE cuSZp_static)
target_link_libraries(${cuSZp_test_f32_3D} PRIVATE cuSZp_static)
target_link_libraries(${cuSZp_test_f64_3D} PRIVATE cuSZp_static)
target_link_libraries(${cuSZp_bench_1D_f32_plain} PRIVATE cuSZp_static)
target_link_libraries(${cuSZp_profile_compress} PRIVATE cuSZp_static)
target_link_libraries(${cuSZp_profile_decompress} PRIVATE cuSZp_static)

# Set output paths for the compiled binary
set_target_properties(${cuSZp} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${cuSZp_test_f32_1D} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${cuSZp_test_f64_1D} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${cuSZp_test_f32_2D} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${cuSZp_test_f64_2D} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${cuSZp_test_f32_3D} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${cuSZp_test_f64_3D} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${cuSZp_bench_1D_f32_plain} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${cuSZp_profile_compress} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${cuSZp_profile_decompress} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})

# Set installation paths for the compiled binary
install(TARGETS ${cuSZp} DESTINATION bin)
install(TARGETS ${cuSZp_test_f32_1D} DESTINATION bin)
install(TARGETS ${cuSZp_test_f64_1D} DESTINATION bin)
install(TARGETS ${cuSZp_test_f32_2D} DESTINATION bin)
install(TARGETS ${cuSZp_test_f64_2D} DESTINATION bin)
install(TARGETS ${cuSZp_test_f32_3D} DESTINATION bin)
install(TARGETS ${cuSZp_test_f64_3D} DESTINATION bin)
install(TARGETS ${cuSZp_bench_1D_f32_plain} DESTINATION bin)
install(TARGETS ${cuSZp_profile_compress} DESTINATION bin)
install(TARGETS ${cuSZp_profile_decompress} DESTINATION bin)