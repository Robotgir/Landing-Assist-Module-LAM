cmake_minimum_required(VERSION 3.10)
project(MySLZDApp)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages.
find_package(yaml-cpp REQUIRED)
find_package(PCL REQUIRED)
find_package(Open3D REQUIRED)

find_package(octomap REQUIRED)  # Sets OCTOMAP_INCLUDE_DIRS & OCTOMAP_LIBRARIES
include_directories(${OCTOMAP_INCLUDE_DIRS})

# Set include directories relative to the current CMakeLists.txt location.
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/hazard_metrices
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/preprocessing
)

# Add your main executable.
add_executable(main main.cpp)

# Link libraries.
target_link_libraries(main
  yaml-cpp 
  Open3D::Open3D 
  ${PCL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}  
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

# Optionally install the executable.
install(TARGETS main DESTINATION bin)
