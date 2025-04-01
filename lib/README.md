Here's a **standardized and easier-to-read version** of your README file:

---

# 📌 3D Point Cloud Processing and Visualization

This project demonstrates how to process and visualize 3D point clouds using:

- **Open3D** – for point cloud processing and visualization.
- **PCL (Point Cloud Library)** – for filtering, downsampling, and advanced operations.
- **OctoMap** – for 3D occupancy mapping.

---

## 🔍 Features

- Load point cloud files
- Downsample using voxel grid filter
- Create 2D and 3D grid maps
- Build KD-Trees and Octrees
- Apply filters:  
  - Statistical Outlier Removal (SOR)  
  - Radius Outlier Removal  
  - Bilateral filter  
- Plane segmentation:
  - RANSAC
  - PROSAC
  - LMEDS
  - Least Squares
- Region growing segmentation
- Visualize using Open3D and PCL
- Convert point clouds to OctoMap
- Rank landing zone candidates using hazard metrics

---

## 📁 Table of Contents

- [Dependencies](#dependencies)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Function Details](#function-details)  
- [Running Tests](#running-tests)  
- [Pipelines](#pipelines)  
- [References](#references)  

---

## 📦 Dependencies

- [Open3D](http://www.open3d.org/)  
- [PCL](http://pointclouds.org/documentation/)  
- [OctoMap](https://octomap.github.io/)  
- C++11 or higher

---

## 📥 Download Sample Point Cloud

Download from:  
📎 [Google Drive](https://drive.google.com/file/d/1gQqce4ZqsO59hb1R1lowTViAK7js4DPa/view?usp=sharing)

---

## ⚙️ Installation

1. **Clone the repo & switch branch:**

```bash
git clone https://github.com/athulkrishnaaei/Landing-Assist-Module-LAM
cd Landing-Assist-Module-LAM
git checkout feature/slzd_library
```

2. **Build the project:**

```bash
cd lib
mkdir build && cd build
sudo cmake ..
sudo make -j
```
---

## ▶️ Usage

To run test cases:

### With visualization:
```bash
./test_pointcloud_preprocessing
```

### Without visualization:
```bash
./test_pointcloud_preprocessing --no-vis
```
Or set in the source:
```cpp
bool g_skipVisualization = true;
```

### Run a specific test:
```bash
./test_pointcloud_preprocessing --gtest_filter=PCLFiltering.ApplyRadiusFilter
```

---

## 🧩 Function Details

### 🔷 Grid-Based Structuring
| Task          | Function | Reference |
|---------------|----------|-----------|
| 3D Grid       | `create_3d_grid()` | [Open3D VoxelGrid](https://www.open3d.org/html/cpp_api/classopen3d_1_1geometry_1_1_voxel_grid.html) |
| 2D Grid       | `create_2d_grid()` | [PCL GridMinimum](https://github.com/PointCloudLibrary/pcl/blob/master/filters/include/pcl/filters/grid_minimum.h) |

### 🔷 Tree-Based Structuring
| Task         | Function | Reference |
|--------------|----------|-----------|
| KD-Tree      | `create_kdtree()` | [Open3D KDTree](https://www.open3d.org/html/cpp_api/classopen3d_1_1geometry_1_1_k_d_tree_flann.html) |
| Octree       | `create_octree()` | [Open3D Octree](https://www.open3d.org/html/cpp_api/classopen3d_1_1geometry_1_1_octree.html) |

### 🔷 Filtering & Downsampling
| Task                  | Function | Reference |
|-----------------------|----------|-----------|
| Voxel Grid Filter     | `apply_voxel_grid_filter()` | [Open3D](https://www.open3d.org/docs/0.11.0/cpp_api/classopen3d_1_1geometry_1_1_point_cloud.html) |
| Statistical Outlier   | `apply_sor_filter()` | [Open3D](https://github.com/isl-org/Open3D/blob/main/cpp/open3d/geometry/PointCloud.cpp#L602) |
| Radius Outlier        | `applyRadiusFilter()` | [PCL Radius Outlier](http://pointclouds.org/documentation/classpcl_1_1_radius_outlier_removal.html) |
| Bilateral Filter      | `applyBilateralFilter()` | [PCL Bilateral](https://pointclouds.org/documentation/classpcl_1_1_bilateral_filter.html) |

---

## 🛬 Hazard Metrics and Plane Detection

| Method | Function | Reference |
|--------|----------|-----------|
| PCA | `PrincipleComponentAnalysis()` | [PCL PCA](https://pointclouds.org/documentation/classpcl_1_1_p_c_a.html) |
| RANSAC (Open3D) | `RansacPlaneSegmentation()` | [Open3D RANSAC](https://github.com/isl-org/Open3D/blob/main/examples/cpp/RegistrationRANSAC.cpp) |
| RANSAC (PCL) | `performRANSAC()` | [PCL RANSAC](https://pointclouds.org/documentation/classpcl_1_1_s_a_c_segmentation.html) |
| PROSAC | `performPROSAC()` | [SAC_PROSAC](https://pcl.readthedocs.io/projects/tutorials/en/latest/planar_segmentation.html) |
| LMEDS | `performLMEDS()` | [SAC_LMEDS](https://pcl.readthedocs.io/projects/tutorials/en/latest/planar_segmentation.html) |
| Least Squares | `LeastSquaresPlaneFitting()` | — |
| Region Growing | `regionGrowingSegmentation()` | [PCL Region Growing](https://pcl.readthedocs.io/projects/tutorials/en/latest/region_growing_segmentation.html) |

---

## 🧪 Running Tests

### Run hazard metric test with visualization:
```bash
./test_hazard_metrices
```

### Without visualization:
```bash
./test_hazard_metrices --no-vis
```

### Run specific test (e.g., PROSAC):
```bash
./test_hazard_metrices --gtest_filter=HazardMetricesTest.TestPROSAC
```

---

## 🔄 YAML-Based Pipeline Execution

1. Navigate to `lib/`, create and enter build folder:
```bash
cd lib
mkdir build && cd build
sudo make
```

2. Modify the YAML file path in `main.cpp`

Available YAML files:

- `allFunctionTest.yaml` – Run full pipeline step-by-step  
- `pipeline1.yaml` – Uses Architecture 1 from the paper  
- `pipeline2Octree.yaml` – Use Architecture 2 from the paper. Grows patches using octree + PCA  
  - Change line 329 in `main.cpp` to switch to KDTree:
    ```cpp
    octreeNeighbourhoodPCAFilterOMP → kdtreeNeighbourhoodPCAFilterOMP
    ```
- `pipeline3.yaml` – Custom region growing (incomplete)


3. **Run the main executable:**

```bash
./main
```
This will run the main executable with the yaml specified in the code,change line 17 in main file to the specified yaml file 
---

## 🏁 Candidate Ranking from PCL Result

Use the function:
```cpp
rankCandidatePatchFromPCLResult(pclResult, hazardMetricsName);
```

To rank based on:

- `"ALL"` – All hazard metrics  
- `"ROUGHNESS"` – Roughness only  
- `"RELIEF"` – Relief only  
- `"DATA_CONFIDENCE"` – Confidence level of data

Example YAML entry:

```yaml
- step: "HazarMetrices"
  enabled: true
  parameters:
    hazard: "ALL"
```

---

## 🔗 References

- [Open3D Documentation](https://www.open3d.org/docs/latest/index.html)
- [PCL Tutorials](https://pcl.readthedocs.io/en/latest/)
- [OctoMap](https://octomap.github.io/)
- [RANSAC in Open3D](https://github.com/isl-org/Open3D/blob/main/examples/cpp/RegistrationRANSAC.cpp)

---
