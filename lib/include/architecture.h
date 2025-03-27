#ifndef ARCHITECUTE_H
#define ARCHITECUTE_H

#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
#include <cmath>
#include <limits>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/lmeds.h>  

#include <pcl/filters/extract_indices.h>

#include <pcl/surface/mls.h>
#include <pcl/features/integral_image_normal.h>

#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/common/common.h> 

#include <eigen3/Eigen/Dense>
#include <open3d/Open3D.h>

#include <pcl/common/pca.h>
#include <pcl/surface/convex_hull.h>

#include <omp.h>

#include <common.h>
#include <variant>

#include <pcl/features/normal_3d_omp.h>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/octree/octree_search.h>

#include<queue>
#include <unordered_set>





// Check if patch is free of obstacles in a clearance region above patchMaxZ
// Helper to compute XY centroid, XY radius, and maximum Z from a patch.
inline void computePatchData(const pcl::PointCloud<PointT>::Ptr &patchCloud,
    double &cx, double &cy,
    double &patchRadius,
    double &maxZ)
{
if (!patchCloud || patchCloud->empty()) {
cx = cy = patchRadius = maxZ = 0.0;
return;
}
// Compute XY centroid
cx = 0.0;
cy = 0.0;
for (const auto &pt : patchCloud->points) {
cx += pt.x;
cy += pt.y;
}
cx /= patchCloud->size();
cy /= patchCloud->size();

// Compute XY radius (max distance to centroid in XY) and maxZ
patchRadius = 0.0;
maxZ = -std::numeric_limits<double>::infinity();
for (const auto &pt : patchCloud->points) {
double dx = pt.x - cx;
double dy = pt.y - cy;
double distXY = std::hypot(dx, dy);
if (distXY > patchRadius)
patchRadius = distXY;
if (pt.z > maxZ)
maxZ = pt.z;
}
}


// We'll sample along z from patchMaxZ up to patchMaxZ+clearance in small steps
inline bool isPatchCollisionFree(
    const pcl::PointCloud<PointT>::Ptr &cloud,
    pcl::KdTreeFLANN<PointT> &kdtree,
    double cx,            // Patch centroid X
    double cy,            // Patch centroid Y
    double patchRadius,   // Patch's XY radius
    double patchMaxZ,     // Patch's highest Z
    double margin,        // Additional margin above patchMaxZ considered an obstacle
    double clearance,     // Total height above patchMaxZ to check
    double step_size)     // Step size in Z
{
    // We'll sample from patchMaxZ in steps of step_size,
    // up to patchMaxZ + clearance
    int steps = static_cast<int>(std::ceil(clearance / step_size));
    
    // The threshold above which we consider it "blocked" if we find a point
    double obstacleZThreshold = patchMaxZ + margin;
    
    for (int i = 0; i <= steps; i++) {
        // Current Z level
        double zCurrent = patchMaxZ + i * step_size;
        
        // Build a search point at this Z level, same XY center
        PointT searchPt;
        searchPt.x = static_cast<float>(cx);
        searchPt.y = static_cast<float>(cy);
        searchPt.z = static_cast<float>(zCurrent);
        
        // Perform a radius search with the same XY radius
        std::vector<int> indices;
        std::vector<float> sqrDists;
        int found = kdtree.radiusSearch(searchPt, static_cast<float>(patchRadius),
                                        indices, sqrDists);
        if (found <= 0) {
            // No points at this level => no immediate obstacle here
            continue;
        }

        // Among these points, check if any have z > patchMaxZ + margin
        // (i.e., actual obstacle above the patch)
        for (int idx : indices) {
            if (cloud->points[idx].z > obstacleZThreshold) {
                // Found an obstacle -> patch is not collision-free
                return false;
            }
        }
    }

    // If we finish the entire sampling without finding obstacles,
    // then the patch is clear in that cylindrical region
    return true;
}


inline std::tuple<PCLResult, SLZDCandidatePoints> octreeNeighborhoodPCAFilter(
    const CloudInput<PointT>& input,
    double initRadius,
    float voxelSize,
    int k,
    float angleThreshold,
    int landingZoneNumber,
    int maxAttempts)
{

    // 1) Load the cloud
    srand(time(0));
    SLZDCandidatePoints finalCandidate;
    std::vector<PCLResult> candidatePatches;

    std::cout << "Loading point cloud..." << std::endl;
    auto cloud = loadPCLCloud<PointT>(input);
    if (!cloud || cloud->empty()) {
        std::cerr << "Error: Loaded point cloud is empty!" << std::endl;
        return { {}, finalCandidate };
    }

    // 2) Compute bounding box (optional: to skip seeds near edges)
    double minX = std::numeric_limits<double>::max(), maxX = -std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max(), maxY = -std::numeric_limits<double>::max();
    double minZ = std::numeric_limits<double>::max(), maxZ = -std::numeric_limits<double>::max();
    for (const auto &pt : cloud->points) {
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
        if (pt.z < minZ) minZ = pt.z;
        if (pt.z > maxZ) maxZ = pt.z;
    }

    std::cout << "Cloud bounding box (XYZ): x=[" << minX << ", " << maxX
              << "], y=[" << minY << ", " << maxY 
              << "], z=[" << minZ << ", " << maxZ << "]\n";

    // 3) Build KdTree
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    // parameters
    int attempts = 0;
    double currentRadius = initRadius;
    double radiusIncrement = 0.5;
    double circularityThreshold = 0.8;

    // for obstacle check
    double clearance = 10.0;  // e.g. how high we check above patch
    double margin = 0.01;     // how far above patchMaxZ to consider an obstacle
    double step_size = 0.2;
    // main loop
    while ((candidatePatches.size() < static_cast<size_t>(landingZoneNumber)) &&
           (attempts < maxAttempts))
    {
        attempts++;
        int randomIndex = rand() % cloud->points.size();
        PointT searchPoint = cloud->points[randomIndex];

        // skip seeds near edges
        if (searchPoint.x < (minX + initRadius) || searchPoint.x > (maxX - initRadius) ||
            searchPoint.y < (minY + initRadius) || searchPoint.y > (maxY - initRadius))
        {
            std::cout << "Attempt " << attempts << ": Seed near edge => skip.\n";
            continue;
        }

        std::cout << "Attempt " << attempts << ": seed= (" << searchPoint.x << ","
                  << searchPoint.y << "," << searchPoint.z << ")\n";

        bool foundFlat = false;
        PCLResult bestFlatPatch;
        double bestCircularity = 0.0;

        // “grow” patch
        while (true)
        {
            std::vector<int> idx;
            std::vector<float> dist;
            int found = kdtree.radiusSearch(searchPoint, currentRadius, idx, dist);
            std::cout << "  radius= " << currentRadius << " => " << found << " neighbors.\n";
            if (found < 3) break;

            // build patch
            pcl::PointCloud<PointT>::Ptr patch(new pcl::PointCloud<PointT>());
            patch->reserve(found);
            for (int i = 0; i < found; i++)
                patch->points.push_back(cloud->points[idx[i]]);

            // do PCA
            PCLResult pcaResult = PrincipleComponentAnalysis(patch, 
                                                             voxelSize, 
                                                             angleThreshold, 
                                                             k);

            // check if flat
            if (pcaResult.outlier_cloud->empty()) {
                // if flat => check circularity
                pcl::ConvexHull<PointT> chull;
                chull.setInputCloud(patch);
                chull.setDimension(2);
                pcl::PointCloud<PointT>::Ptr hull(new pcl::PointCloud<PointT>());
                chull.reconstruct(*hull);

                double area=0, perimeter=0;
                if (hull->size() >=3) {
                    for (size_t i=0; i < hull->size(); i++){
                        size_t j = (i+1)%hull->size();
                        double xi= hull->points[i].x, yi= hull->points[i].y;
                        double xj= hull->points[j].x, yj= hull->points[j].y;
                        area += (xi*yj - xj*yi);
                        perimeter += std::hypot(xj - xi, yj - yi);
                    }
                    area = std::fabs(area)*0.5;
                }
                double circ = (perimeter > 0)? (4.0*M_PI*area)/(perimeter*perimeter) : 0.0;
                std::cout << "    => flat, circ= " << circ << "\n";

                if (circ >= circularityThreshold) {
                    // so far so good => keep track
                    bestFlatPatch = pcaResult;
                    bestCircularity= circ;
                    foundFlat= true;
                }
                // else it's flat but not “circular enough,” keep going or break is your choice

                // Grow radius
                currentRadius += radiusIncrement;
                continue;
            }
            else {
                std::cout << "    => not flat => stop growing.\n";
            }
            break;
        } // end while “grow"
        // Print number of candidate patches found so far (before obstacle check).
        std::cout << "Candidate patches found so far (before collision check): " << candidatePatches.size() << "\n";
        // if we ended with foundFlat => do an obstacle check
        if (foundFlat) {
            // compute patch’s XY center, radius, maxZ
            double cx=0, cy=0, patchR=0, patchMaxZ=0;
            computePatchData(bestFlatPatch.inlier_cloud, cx, cy, patchR, patchMaxZ);

            // check if collision-free
            bool collisionFree = isPatchCollisionFree(cloud, kdtree,
                                                      cx, cy,
                                                      patchR,
                                                      patchMaxZ,
                                                      margin,    // how high above patch we consider obstacle
                                                      clearance,  // how far above patchMaxZ we check
                                                      step_size);
            
            if (collisionFree) {
                std::cout << "Candidate patch is collision-free above => accepting.\n";
                candidatePatches.push_back(bestFlatPatch);
                finalCandidate.seedPoints.push_back(searchPoint);
                finalCandidate.plane_coefficients.push_back(bestFlatPatch.plane_coefficients);
            }
            else {
                std::cout << "Candidate patch has obstacles above => skip.\n";
            }
        }
        else {
            std::cout << "No suitable patch from this seed.\n";
        }

        // reset radius for next seed
        currentRadius = initRadius;
    } // end while attempts
    std::cout << "Loop ended with attempts = " << attempts 
          << ", found " << candidatePatches.size() << " patches.\n";

    // Merge candidate patches
    PCLResult finalResult;
    finalResult.inlier_cloud.reset(new pcl::PointCloud<PointT>());
    finalResult.outlier_cloud.reset(new pcl::PointCloud<PointT>());
    finalResult.downsampled_cloud = cloud;

    for (auto &patch : candidatePatches) {
        finalResult.inlier_cloud->insert(finalResult.inlier_cloud->end(),
                                         patch.inlier_cloud->begin(),
                                         patch.inlier_cloud->end());
        finalCandidate.detectedSurfaces.push_back(patch.inlier_cloud);
    }
    // Optionally fill outlier with all points
    for (size_t i=0; i<cloud->size(); i++)
        finalResult.outlier_cloud->push_back(cloud->points[i]);

    std::cout << "Found " << candidatePatches.size() << " candidate patches.\n";
    std::cout << "Inlier patch= " << finalResult.inlier_cloud->size() << " points.\n";
    std::cout << "Outlier= " << finalResult.outlier_cloud->size() << " points.\n";

    return { finalResult, finalCandidate };
}


//=====================================================================================================================================================================
inline std::vector<SLZDCandidatePoints> rankCandidatePoints(
    std::vector<SLZDCandidatePoints>& candidatePoints, 
    PCLResult& result) 
{
    // For each candidate, compute metrics and individual surface scores
    for (auto& candidate : candidatePoints) {
        // Clear previous metrics
        candidate.dataConfidences.clear();
        candidate.reliefs.clear();
        candidate.roughnesses.clear();
        candidate.score.clear();
        
        std::vector<double> individualScores;  // Temporary vector for individual surface scores

        // Iterate through each detected surface
        for (size_t i = 0; i < candidate.detectedSurfaces.size(); ++i) {
            PCLResult surfResult;
            surfResult.inlier_cloud = candidate.detectedSurfaces[i];
            surfResult.plane_coefficients = candidate.plane_coefficients[i];

            // Calculate metrics for this surface
            double dataConfidence = calculateDataConfidencePCL(surfResult);
            candidate.dataConfidences.push_back(dataConfidence);

            double relief = calculateReliefPCL(surfResult);
            candidate.reliefs.push_back(relief);

            double roughness = calculateRoughnessPCL(surfResult);
            candidate.roughnesses.push_back(roughness);

            // Compute the individual surface score: higher confidence, lower relief and roughness are preferred
            double surfaceScore = dataConfidence - relief - roughness;
            individualScores.push_back(surfaceScore);
        }
        
        // Store the individual surface scores in candidate.score
        candidate.score = individualScores;
        
        // Compute the final candidate score as the average of the individual surface scores
        double total = 0.0;
        for (double s : individualScores) {
            total += s;
        }
        double averageScore = individualScores.empty() ? 0.0 : total / individualScores.size();
        
        // Clear individual scores and append only the final average score if you want to use it for sorting.
        // (Alternatively, you can store both if you wish.)
        candidate.score.clear();
        candidate.score.push_back(averageScore);
    }
    
    // Sort candidate patches in descending order based on their final average score
    std::sort(candidatePoints.begin(), candidatePoints.end(), 
              [](const SLZDCandidatePoints& a, const SLZDCandidatePoints& b) {
                  double finalA = a.score.empty() ? 0.0 : a.score.back();
                  double finalB = b.score.empty() ? 0.0 : b.score.back();
                  return finalA > finalB; // Descending order: higher score is better
              }
    );
    
    // Print detailed information for each candidate patch
    std::cout << "Candidate Patches Details (sorted in descending order):" << std::endl;
    for (size_t idx = 0; idx < candidatePoints.size(); idx++) {
         const auto& candidate = candidatePoints[idx];
         double finalScore = candidate.score.empty() ? 0.0 : candidate.score.back();
         std::cout << "Candidate " << idx + 1 << ":" << std::endl;
         std::cout << "  Final Average Score: " << finalScore << std::endl;
         
         std::cout << "  Individual Surface Scores:" << std::endl;
         // Note: If you cleared individual scores and only stored the average,
         // you might want to print the stored data confidences, reliefs, and roughnesses instead.
         for (size_t i = 0; i < candidate.dataConfidences.size(); i++) {
            std::cout << "    Surface " << i + 1 << ":" << std::endl;
            std::cout << "      Data Confidence: " << candidate.dataConfidences[i] << std::endl;
            std::cout << "      Relief: " << candidate.reliefs[i] << std::endl;
            std::cout << "      Roughness: " << candidate.roughnesses[i] << std::endl;
         }
         
         std::cout << "  Seed Points:" << std::endl;
         for (const auto& pt : candidate.seedPoints) {
             std::cout << "    (" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
         }
         
         std::cout << "  Plane Coefficients per Surface:" << std::endl;
         for (size_t i = 0; i < candidate.plane_coefficients.size(); i++) {
             const auto& coeffs = candidate.plane_coefficients[i]->values;
             std::cout << "    Surface " << i + 1 << ": ";
             if (coeffs.size() >= 4) {
                 std::cout << "A=" << coeffs[0] << ", B=" << coeffs[1]
                           << ", C=" << coeffs[2] << ", D=" << coeffs[3];
             }
             std::cout << std::endl;
         }
         std::cout << "-------------------------------------------------" << std::endl;
    }
    
    return candidatePoints;
}

//=========================================================================================================================================================
// working normaly without obstaccle filtering 
// inline std::tuple<PCLResult, SLZDCandidatePoints> octreeNeighborhoodPCAFilter(
//     const CloudInput<PointT>& input,
//     double initRadius,
//     float voxelSize,
//     int k,
//     float angleThreshold,
//     int landingZoneNumber,
//     int maxAttempts)
// {
//     // Seed random generator for unique results.
//     srand(time(0));

//     SLZDCandidatePoints finalCandidate;  // This will store candidate summary info.
//     std::vector<PCLResult> candidatePatches; // Container for candidate patches.

//     std::cout << "Loading point cloud..." << std::endl;
//     auto cloud = loadPCLCloud<PointT>(input);
//     if (!cloud || cloud->points.empty()) {
//         std::cerr << "Error: Loaded point cloud is empty!" << std::endl;
//         PCLResult emptyResult;
//         return std::make_tuple(emptyResult, finalCandidate);
//     }

//     // Compute the 2D bounding box (XY) for the entire cloud.
//     double minX = std::numeric_limits<double>::max(), maxX = -std::numeric_limits<double>::max();
//     double minY = std::numeric_limits<double>::max(), maxY = -std::numeric_limits<double>::max();
//     double minZ = std::numeric_limits<double>::max(), maxZ = -std::numeric_limits<double>::max();
    
//     for (const auto &pt : cloud->points) {
//         if (pt.x < minX) minX = pt.x;
//         if (pt.x > maxX) maxX = pt.x;
//         if (pt.y < minY) minY = pt.y;
//         if (pt.y > maxY) maxY = pt.y;
//         if (pt.z < minZ) minZ = pt.z;
//         if (pt.z > maxZ) maxZ = pt.z;
//     }
//     std::cout << "Cloud bounding box (XYZ): x=[" << minX << ", " << maxX 
//               << "], y=[" << minY << ", " << maxY << "]"
//               << "], z=[" << minZ << ", " << maxZ << "]" << std::endl;

//     // Build KD-tree for the entire cloud.
//     pcl::KdTreeFLANN<PointT> kdtree;
//     kdtree.setInputCloud(cloud);

//     std::unordered_set<int> acceptedIndices;  // Optionally track accepted indices.
//     int attempts = 0;
//     double currentRadius = initRadius;
//     double radiusIncrement = 0.5;           // Increase radius by 0.5 m each time.
//     double circularityThreshold = 0.8;        // Minimum circularity for a full circle.

//     // Optionally define a height filter if needed:
//     double z_min_candidate = minZ;
//     double z_max_candidate = maxZ;

//     while (candidatePatches.size() < static_cast<size_t>(landingZoneNumber) && attempts < maxAttempts) {
//         attempts++;
//         int randomIndex = rand() % cloud->points.size();
//         PointT searchPoint = (*cloud)[randomIndex];

//         // Check if the random point is away from the edge.
//         if (searchPoint.x < (minX + initRadius) || searchPoint.x > (maxX - initRadius) ||
//             searchPoint.y < (minY + initRadius) || searchPoint.y > (maxY - initRadius)) {
//             std::cout << "Attempt " << attempts << ": Random point at (" << searchPoint.x << ", " 
//                       << searchPoint.y << ", " << searchPoint.z 
//                       << ") is too close to the edge. Skipping." << std::endl;
//             continue;
//         }

//         std::cout << "Attempt " << attempts << ": Searching neighbors at ("
//                   << searchPoint.x << " " << searchPoint.y << " " << searchPoint.z
//                   << ") with initial radius = " << currentRadius << " m" << std::endl;

//         bool foundFlat = false;
//         PCLResult bestFlatPatch;  // To store the best flat patch before it stops being flat.
//         double bestCircularity = 0.0;

//         // Grow the patch until it stops being flat.
//         while (true) {
//             std::vector<int> pointIdxRadiusSearch;
//             std::vector<float> pointRadiusSquaredDistance;
//             int neighborsFound = kdtree.radiusSearch(searchPoint, currentRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance);

//             std::cout << "   Radius = " << currentRadius << " m: Found " << neighborsFound << " neighbors." << std::endl;

//             if (neighborsFound <= 0) {
//                 break;  // No neighbors found at this radius.
//             }

//             // Build the patch cloud.
//             typename pcl::PointCloud<PointT>::Ptr patchCloud(new pcl::PointCloud<PointT>());
//             for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
//                 // If you need height filtering, add conditions here.
//                 patchCloud->points.push_back(cloud->points[pointIdxRadiusSearch[i]]);
//             }

//             if (patchCloud->points.size() < 3) {
//                 break;
//             }

//             // Apply PCA to the patch.
//             PCLResult pcaResult = PrincipleComponentAnalysis(patchCloud, voxelSize, angleThreshold, k);

//             // If patch is flat (i.e. no outlier points), check its circularity.
//             if (pcaResult.outlier_cloud->points.empty()) {
//                 pcl::ConvexHull<PointT> chull;
//                 chull.setInputCloud(patchCloud);
//                 chull.setDimension(2);
//                 typename pcl::PointCloud<PointT>::Ptr hull(new pcl::PointCloud<PointT>());
//                 chull.reconstruct(*hull);

//                 double area = 0.0, perimeter = 0.0;
//                 if (hull->points.size() >= 3) {
//                     for (size_t i = 0; i < hull->points.size(); ++i) {
//                         size_t j = (i + 1) % hull->points.size();
//                         double xi = hull->points[i].x, yi = hull->points[i].y;
//                         double xj = hull->points[j].x, yj = hull->points[j].y;
//                         area += (xi * yj - xj * yi);
//                         perimeter += std::hypot(xj - xi, yj - yi);
//                     }
//                     area = std::abs(area) * 0.5;
//                 }
//                 double circularity = (perimeter > 0) ? (4 * M_PI * area) / (perimeter * perimeter) : 0.0;
//                 std::cout << "   Patch is flat. Circularity = " << circularity << std::endl;

//                 if (circularity >= circularityThreshold) {
//                     std::cout << "   Patch is flat and circular at radius " << currentRadius << " m." << std::endl;
//                     bestFlatPatch = pcaResult;
//                     bestCircularity = circularity;
//                     foundFlat = true;
//                 } else {
//                     std::cout << "   Patch is flat but incomplete (circularity = " << circularity << ")." << std::endl;
//                 }
//                 // Increase the radius to try to grow a larger patch.
//                 currentRadius += radiusIncrement;
//                 continue;
//             } else {
//                 std::cout << "   Patch is not flat at radius " << currentRadius << " m." << std::endl;
//                 // Stop growing if patch becomes non-flat.
//                 break;
//             }
//         }  // End inner while

//         // Use the best flat patch from before the patch stopped growing.
//         if (foundFlat) {
//             std::cout << "Saving candidate patch from previous flat search (radius < " 
//                       << currentRadius << " m, circularity = " << bestCircularity << ")." << std::endl;
//             // Visualize the first candidate patch (if none have been saved yet).
//             // if (candidatePatches.empty()) {
//             //     std::cout << "Visualizing first candidate patch..." << std::endl;
//             //     visualizePCL(bestFlatPatch, "both");
//             // }
//             finalCandidate.seedPoints.push_back(searchPoint);
//             candidatePatches.push_back(bestFlatPatch);
//             finalCandidate.plane_coefficients.push_back(bestFlatPatch.plane_coefficients);
//         } else {
//             std::cout << "No suitable full flat patch found in this attempt." << std::endl;
//         }

//         // Reset currentRadius for the next candidate.
//         currentRadius = initRadius;
//     }  // End main while

//     // Merge candidate patches into a single inlier cloud.
//     PCLResult finalResult;
//     finalResult.inlier_cloud = pcl::make_shared<PointCloudT>();
//     finalResult.outlier_cloud = pcl::make_shared<PointCloudT>();
//     finalResult.downsampled_cloud = cloud;  // Keep the original cloud.

//     for (const auto& patch : candidatePatches) {
//         finalResult.inlier_cloud->points.insert(finalResult.inlier_cloud->points.end(),
//                                                   patch.inlier_cloud->points.begin(),
//                                                   patch.inlier_cloud->points.end());
//         finalCandidate.detectedSurfaces.push_back(patch.inlier_cloud);
//     }

//     // Optionally, add all points from the original cloud to the outlier cloud.
//     for (size_t i = 0; i < cloud->points.size(); ++i) {
//         finalResult.outlier_cloud->points.push_back(cloud->points[i]);
//     }

//     std::cout << "Found " << candidatePatches.size() 
//               << " candidate landing zones after " << maxAttempts << " attempts." << std::endl;
//     std::cout << "Final inlier cloud size (candidate patches): " << finalResult.inlier_cloud->points.size() << " points." << std::endl;
//     std::cout << "Final outlier cloud size (remaining points): " << finalResult.outlier_cloud->points.size() << " points." << std::endl;

//     return std::make_tuple(finalResult, finalCandidate);
// }
//===================================================================================================================================================================================



#endif