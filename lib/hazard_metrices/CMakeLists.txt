cmake_minimum_required(VERSION 3.10)
project(Preprocessing)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------------
# 1. Find OPEN3d,PCL and include the necessary modules
#    (Add 'io' if you need loadPCDFile())
# ----------------------------------------------------------------------------

find_package(Open3D REQUIRED)

find_package(PCL 1.4 REQUIRED COMPONENTS common filters io visualization segmentation surface integral_image_normal region_growing)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(octomap REQUIRED)  # This will set OCTOMAP_INCLUDE_DIRS & OCTOMAP_LIBRARIES
include_directories(${OCTOMAP_INCLUDE_DIRS})
# add_definitions(${OCTOMAP_DEFINITIONS})

# ----------------------------------------------------------------------------
# 2. Optionally, find OpenMP if needed for parallelization
# ----------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    # Let CMake automatically add the correct compile/link flags
    # for OpenMP instead of manually using -fopenmp or -lgomp
    # (This is the recommended modern approach)
    message(STATUS "OpenMP found. Linking to OpenMP libraries.")
endif()

# ----------------------------------------------------------------------------
# 3. Manually include other libraries' include paths
#    (Only do this if you do NOT rely on pkg-config or CMake find_package
#     for these libraries)
# ----------------------------------------------------------------------------
include_directories(
   
    /usr/local/include/eigen3
    /usr/local/include/pcl-1.14

)
  
# ----------------------------------------------------------------------------
# 4. Create the executable
# ----------------------------------------------------------------------------
add_executable(main main.cpp)

# ----------------------------------------------------------------------------
# 5. Link libraries
#    - PCL libraries (from find_package)
#    - OpenMP (if found)
#    - Other manually specified libraries
# ----------------------------------------------------------------------------

 
target_link_libraries(main
    PRIVATE
    Open3D::Open3D                  # -lOpen3D
    ${PCL_LIBRARIES} 
    ${OCTOMAP_LIBRARIES}  
    octomap 
    octomath
    GL                     # -lGL
    GLU                    # -lGLU
    glfw                   # -lglfw
    GLEW                   # -lGLEW
    assimp                 # -lassimp
    jsoncpp               # -ljsoncpp
    jpeg                   # -ljpeg
    png                    # -lpng
    z                      # -lz
    pthread                # -lpthread
)

add_executable(test_hazard_metrices test_hazard_metrices.cpp)
target_link_libraries(test_hazard_metrices 
    ${PCL_LIBRARIES} 
    Open3D::Open3D 
    octomap 
    gtest 
    gtest_main
    GL                     # -lGL
    GLU                    # -lGLU
    glfw                   # -lglfw
    GLEW                   # -lGLEW
    assimp                 # -lassimp
    jsoncpp               # -ljsoncpp
    jpeg                   # -ljpeg
    png                    # -lpng
    z                      # -lz
    pthread                # -lpthread
)

# Link OpenMP last to ensure correct symbol resolution
if(OPENMP_FOUND)
    target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
endif()
