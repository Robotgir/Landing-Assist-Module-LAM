cmake_minimum_required(VERSION 3.10)
project(MySLZDApp)

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
add_executable(MySLZDApp main.cpp)

# Link libraries.
target_link_libraries(MySLZDApp 
  yaml-cpp 
  Open3D::Open3D 
  ${PCL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}  
)

# Optionally install the executable.
install(TARGETS MySLZDApp DESTINATION bin)
