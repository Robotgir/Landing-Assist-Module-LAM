#ifndef ARCHITECTURE_H
#define ARCHITECTURE_H

#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cstdlib>
#include <ctime>

// PCL includes
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
#include <pcl/common/pca.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/octree/octree_search.h>

// Possibly for obstacle checking or other tasks
#include <open3d/Open3D.h>
#include "common.h"  // For PCLResult, loadPCLCloud, etc.


//-----------------------------------------------------------------------------
// computePatchData: compute XY centroid, patchRadius, and maxZ for a given patch
//-----------------------------------------------------------------------------
inline void computePatchData(const pcl::PointCloud<pcl::PointXYZI>::Ptr &patchCloud,
                             double &cx, double &cy,
                             double &patchRadius,
                             double &maxZ)
{
    if (!patchCloud || patchCloud->empty()) {
        cx = cy = patchRadius = maxZ = 0.0;
        return;
    }
    // Compute XY centroid
    cx = 0.0; cy = 0.0;
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
        if (distXY > patchRadius) {
            patchRadius = distXY;
        }
        if (pt.z > maxZ) {
            maxZ = pt.z;
        }
    }
}

//-----------------------------------------------------------------------------
// isPatchCollisionFreeOctree: checks if there's an obstacle above the patch
//-----------------------------------------------------------------------------
inline bool isPatchCollisionFreeOctree(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZI> &octree,  
    double cx,            
    double cy,            
    double patchRadius,   
    double patchMaxZ,     
    double margin,        
    double clearance,     
    double step_size)     
{
    int steps = static_cast<int>(std::ceil(clearance / step_size));
    double obstacleZThreshold = patchMaxZ + margin;

    for (int i = 0; i <= steps; i++) {
        double zCurrent = patchMaxZ + i * step_size;
        pcl::PointXYZI searchPt;
        searchPt.x = static_cast<float>(cx);
        searchPt.y = static_cast<float>(cy);
        searchPt.z = static_cast<float>(zCurrent);
        
        std::vector<int> indices;
        std::vector<float> sqrDists;
        int found = octree.radiusSearch(searchPt, static_cast<float>(patchRadius),
                                        indices, sqrDists);
        if (found <= 0) {
            continue;
        }
        for (int idx : indices) {
            if (cloud->points[idx].z > obstacleZThreshold) {
                return false; // Found an obstacle
            }
        }
    }
    return true;
}

//-----------------------------------------------------------------------------
// octreeNeighborhoodPCAFilter: returns multiple collision-free patches
//-----------------------------------------------------------------------------
inline std::tuple<PCLResult, std::vector<SLZDCandidatePoints>>
octreeNeighborhoodPCAFilter(
    const CloudInput<pcl::PointXYZI>& input,
    double initRadius,
    float voxelSize,
    int k,
    float angleThreshold,
    int landingZoneNumber,
    int maxAttempts)
{
    srand(static_cast<unsigned int>(time(0)));

    // We'll store all accepted patches in finalCandidates
    std::vector<SLZDCandidatePoints> finalCandidates;  
    std::vector<PCLResult> candidatePatches; // For demonstration

    std::cout << "Loading point cloud..." << std::endl;
    auto cloud = loadPCLCloud<pcl::PointXYZI>(input);
    if (!cloud || cloud->empty()) {
        std::cerr << "Error: Loaded point cloud is empty!" << std::endl;
        return std::make_tuple(PCLResult(), finalCandidates);
    }

    // Compute bounding box to skip seeds near edges
    double minX = std::numeric_limits<double>::max(), maxX = -std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max(), maxY = -std::numeric_limits<double>::max();
    for (const auto &pt : cloud->points) {
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
    }

    double minZ = std::numeric_limits<double>::max(), maxZ = -std::numeric_limits<double>::max();
    for (const auto &pt : cloud->points) {
        if (pt.z < minZ) minZ = pt.z;
        if (pt.z > maxZ) maxZ = pt.z;
    }

    std::cout << "Cloud bounding box (XYZ): x=[" << minX << ", " << maxX
              << "], y=[" << minY << ", " << maxY
              << "], z=[" << minZ << ", " << maxZ << "]\n";

    // Build an octree
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZI> octree(0.001);
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    int attempts = 0;
    double currentRadius = initRadius;
    double radiusIncrement = 0.5;
    double circularityThreshold = 0.8;
    // Obstacle check parameters
    double clearance = 10.0;
    double margin = 0.01;
    double step_size = 0.2;

    while (finalCandidates.size() < static_cast<size_t>(landingZoneNumber) &&
           attempts < maxAttempts)
    {
        attempts++;
        int randomIndex = rand() % cloud->size();
        pcl::PointXYZI searchPoint = cloud->points[randomIndex];

        // skip seeds near edges
        if (searchPoint.x < (minX + initRadius) || searchPoint.x > (maxX - initRadius) ||
            searchPoint.y < (minY + initRadius) || searchPoint.y > (maxY - initRadius))
        {
            std::cout << "Attempt " << attempts << ": Seed near edge => skip.\n";
            continue;
        }

        std::cout << "Attempt " << attempts << ": seed= ("
                  << searchPoint.x << ", " << searchPoint.y << ", " << searchPoint.z
                  << ")\n";

        bool foundFlat = false;
        PCLResult bestFlatPatch;
        double bestCircularity = 0.0;

        // "Grow" patch
        while (true) {
            std::vector<int> idx;
            std::vector<float> dist;
            int found = octree.radiusSearch(searchPoint, currentRadius, idx, dist);
            std::cout << "  radius= " << currentRadius << " => " << found << " neighbors.\n";
            if (found < 3) break;

            // Build patch cloud
            pcl::PointCloud<pcl::PointXYZI>::Ptr patch(new pcl::PointCloud<pcl::PointXYZI>());
            patch->reserve(found);
            for (int i = 0; i < found; i++) {
                patch->points.push_back(cloud->points[idx[i]]);
            }

            // Apply your PCA-based flattening check
            PCLResult pcaResult = PrincipleComponentAnalysis(patch, voxelSize, angleThreshold, k);

            if (pcaResult.outlier_cloud->empty()) {
                // Patch is flat.  Check circularity
                pcl::ConvexHull<pcl::PointXYZI> chull;
                chull.setInputCloud(patch);
                chull.setDimension(2);

                pcl::PointCloud<pcl::PointXYZI>::Ptr hull(new pcl::PointCloud<pcl::PointXYZI>());
                chull.reconstruct(*hull);

                double area = 0.0, perimeter = 0.0;
                if (hull->size() >= 3) {
                    for (size_t i = 0; i < hull->size(); i++) {
                        size_t j = (i + 1) % hull->size();
                        double xi = hull->points[i].x, yi = hull->points[i].y;
                        double xj = hull->points[j].x, yj = hull->points[j].y;
                        area += (xi * yj - xj * yi);
                        perimeter += std::hypot(xj - xi, yj - yi);
                    }
                    area = std::fabs(area) * 0.5;
                }
                double circ = (perimeter > 0) ? (4.0 * M_PI * area) / (perimeter * perimeter) : 0.0;
                std::cout << "    => flat, circ= " << circ << "\n";
                if (circ >= circularityThreshold) {
                    bestFlatPatch = pcaResult;
                    bestCircularity = circ;
                    foundFlat = true;
                }
                currentRadius += radiusIncrement;
                continue;
            } else {
                std::cout << "    => not flat => stop growing.\n";
            }
            break;
        } // end while "grow"

        std::cout << "Candidate patches found so far (before obstacle check): " << finalCandidates.size() << "\n";

        if (foundFlat) {
            // compute patch data
            double cx = 0, cy = 0, patchR = 0, patchMaxZ = 0;
            computePatchData(bestFlatPatch.inlier_cloud, cx, cy, patchR, patchMaxZ);

            bool collisionFree = isPatchCollisionFreeOctree(
                cloud, octree, cx, cy, patchR, patchMaxZ,
                margin, clearance, step_size);

            if (collisionFree) {
                std::cout << "Candidate patch is collision-free => accepting.\n";

                // Create a new candidate object
                SLZDCandidatePoints candidate;
                candidate.seedPoint.x = searchPoint.x;
                candidate.seedPoint.y = searchPoint.y;
                candidate.seedPoint.z = searchPoint.z;
                candidate.plane_coefficients = bestFlatPatch.plane_coefficients;

                // Metric calculations
                candidate.dataConfidence = calculateDataConfidencePCL(bestFlatPatch);
                candidate.relief         = calculateReliefPCL(bestFlatPatch);
                candidate.roughness      = calculateRoughnessPCL(bestFlatPatch);
                // Score
                candidate.score = candidate.dataConfidence
                                 - candidate.relief
                                 - candidate.roughness;
                // Detected surface
                candidate.detectedSurface = bestFlatPatch.inlier_cloud;

                // Save this candidate in our vector
                finalCandidates.push_back(candidate);

                // We also add this patch to candidatePatches for merging into finalResult.
                candidatePatches.push_back(bestFlatPatch);
            } else {
                std::cout << "Candidate patch has obstacles => skip.\n";
            }
        } else {
            std::cout << "No suitable patch from this seed.\n";
        }
        currentRadius = initRadius;
    } // end while attempts

    std::cout << "Loop ended with attempts = " << attempts
              << ", found " << finalCandidates.size() << " patches.\n";

    // Build a finalResult from all the found patches
    PCLResult finalResult;
    finalResult.inlier_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    finalResult.outlier_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    finalResult.downsampled_cloud = cloud;

    // Merge them
    for (auto &patch : candidatePatches) {
        finalResult.inlier_cloud->insert(finalResult.inlier_cloud->end(),
                                         patch.inlier_cloud->begin(),
                                         patch.inlier_cloud->end());
    }
    // Optionally add all original points to outlier
    for (size_t i = 0; i < cloud->size(); i++) {
        finalResult.outlier_cloud->push_back(cloud->points[i]);
    }

    std::cout << "Found " << finalCandidates.size() << " candidate patches.\n";
    std::cout << "Inlier patch= " << finalResult.inlier_cloud->size() << " points.\n";
    std::cout << "Outlier= " << finalResult.outlier_cloud->size() << " points.\n";

    // Return a tuple: (finalResult, vectorOfCandidates)
    return std::make_tuple(finalResult, finalCandidates);
}

//-----------------------------------------------------------------------------
// isPatchCollisionFreekdtree: checks if there's an obstacle above the patch
//-----------------------------------------------------------------------------
inline bool isPatchCollisionFreeKdtree(
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
//-----------------------------------------------------------------------------
// kdtreeNeighborhoodPCAFilter: returns multiple collision-free patches
//-----------------------------------------------------------------------------
inline std::tuple<PCLResult, std::vector<SLZDCandidatePoints>>
kdtreeNeighborhoodPCAFilter(
    const CloudInput<pcl::PointXYZI>& input,
    double initRadius,
    float voxelSize,
    int k,
    float angleThreshold,
    int landingZoneNumber,
    int maxAttempts)
{
    srand(static_cast<unsigned int>(time(0)));

    // We'll store all accepted patches in finalCandidates
    std::vector<SLZDCandidatePoints> finalCandidates;  
    std::vector<PCLResult> candidatePatches; // For demonstration

    std::cout << "Loading point cloud..." << std::endl;
    auto cloud = loadPCLCloud<pcl::PointXYZI>(input);
    if (!cloud || cloud->empty()) {
        std::cerr << "Error: Loaded point cloud is empty!" << std::endl;
        return std::make_tuple(PCLResult(), finalCandidates);
    }

    // Compute bounding box to skip seeds near edges
    double minX = std::numeric_limits<double>::max(), maxX = -std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max(), maxY = -std::numeric_limits<double>::max();
    for (const auto &pt : cloud->points) {
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
    }

    double minZ = std::numeric_limits<double>::max(), maxZ = -std::numeric_limits<double>::max();
    for (const auto &pt : cloud->points) {
        if (pt.z < minZ) minZ = pt.z;
        if (pt.z > maxZ) maxZ = pt.z;
    }

    std::cout << "Cloud bounding box (XYZ): x=[" << minX << ", " << maxX
              << "], y=[" << minY << ", " << maxY
              << "], z=[" << minZ << ", " << maxZ << "]\n";

    // Build an octree
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);


    int attempts = 0;
    double currentRadius = initRadius;
    double radiusIncrement = 0.5;
    double circularityThreshold = 0.8;
    // Obstacle check parameters
    double clearance = 10.0;
    double margin = 0.01;
    double step_size = 0.2;

    while (finalCandidates.size() < static_cast<size_t>(landingZoneNumber) &&
           attempts < maxAttempts)
    {
        attempts++;
        int randomIndex = rand() % cloud->size();
        pcl::PointXYZI searchPoint = cloud->points[randomIndex];

        // skip seeds near edges
        if (searchPoint.x < (minX + initRadius) || searchPoint.x > (maxX - initRadius) ||
            searchPoint.y < (minY + initRadius) || searchPoint.y > (maxY - initRadius))
        {
            std::cout << "Attempt " << attempts << ": Seed near edge => skip.\n";
            continue;
        }

        std::cout << "Attempt " << attempts << ": seed= ("
                  << searchPoint.x << ", " << searchPoint.y << ", " << searchPoint.z
                  << ")\n";

        bool foundFlat = false;
        PCLResult bestFlatPatch;
        double bestCircularity = 0.0;

        // "Grow" patch
        while (true) {
            std::vector<int> idx;
            std::vector<float> dist;
            int found = kdtree.radiusSearch(searchPoint, currentRadius, idx, dist);
            std::cout << "  radius= " << currentRadius << " => " << found << " neighbors.\n";
            if (found < 3) break;

            // Build patch cloud
            pcl::PointCloud<pcl::PointXYZI>::Ptr patch(new pcl::PointCloud<pcl::PointXYZI>());
            patch->reserve(found);
            for (int i = 0; i < found; i++) {
                patch->points.push_back(cloud->points[idx[i]]);
            }

            // Apply your PCA-based flattening check
            PCLResult pcaResult = PrincipleComponentAnalysis(patch, voxelSize, angleThreshold, k);

            if (pcaResult.outlier_cloud->empty()) {
                // Patch is flat.  Check circularity
                pcl::ConvexHull<pcl::PointXYZI> chull;
                chull.setInputCloud(patch);
                chull.setDimension(2);

                pcl::PointCloud<pcl::PointXYZI>::Ptr hull(new pcl::PointCloud<pcl::PointXYZI>());
                chull.reconstruct(*hull);

                double area = 0.0, perimeter = 0.0;
                if (hull->size() >= 3) {
                    for (size_t i = 0; i < hull->size(); i++) {
                        size_t j = (i + 1) % hull->size();
                        double xi = hull->points[i].x, yi = hull->points[i].y;
                        double xj = hull->points[j].x, yj = hull->points[j].y;
                        area += (xi * yj - xj * yi);
                        perimeter += std::hypot(xj - xi, yj - yi);
                    }
                    area = std::fabs(area) * 0.5;
                }
                double circ = (perimeter > 0) ? (4.0 * M_PI * area) / (perimeter * perimeter) : 0.0;
                std::cout << "    => flat, circ= " << circ << "\n";
                if (circ >= circularityThreshold) {
                    bestFlatPatch = pcaResult;
                    bestCircularity = circ;
                    foundFlat = true;
                }
                currentRadius += radiusIncrement;
                continue;
            } else {
                std::cout << "    => not flat => stop growing.\n";
            }
            break;
        } // end while "grow"

        std::cout << "Candidate patches found so far (before obstacle check): " << finalCandidates.size() << "\n";

        if (foundFlat) {
            // compute patch data
            double cx = 0, cy = 0, patchR = 0, patchMaxZ = 0;
            computePatchData(bestFlatPatch.inlier_cloud, cx, cy, patchR, patchMaxZ);

            bool collisionFree = isPatchCollisionFreeKdtree(
                cloud, kdtree, cx, cy, patchR, patchMaxZ,
                margin, clearance, step_size);

            if (collisionFree) {
                std::cout << "Candidate patch is collision-free => accepting.\n";

                // Create a new candidate object
                SLZDCandidatePoints candidate;
                candidate.seedPoint.x = searchPoint.x;
                candidate.seedPoint.y = searchPoint.y;
                candidate.seedPoint.z = searchPoint.z;
                candidate.plane_coefficients = bestFlatPatch.plane_coefficients;

                // Metric calculations
                candidate.dataConfidence = calculateDataConfidencePCL(bestFlatPatch);
                candidate.relief         = calculateReliefPCL(bestFlatPatch);
                candidate.roughness      = calculateRoughnessPCL(bestFlatPatch);
                // Score
                candidate.score = candidate.dataConfidence
                                 - candidate.relief
                                 - candidate.roughness;
                // Detected surface
                candidate.detectedSurface = bestFlatPatch.inlier_cloud;

                // Save this candidate in our vector
                finalCandidates.push_back(candidate);

                // We also add this patch to candidatePatches for merging into finalResult.
                candidatePatches.push_back(bestFlatPatch);
            } else {
                std::cout << "Candidate patch has obstacles => skip.\n";
            }
        } else {
            std::cout << "No suitable patch from this seed.\n";
        }
        currentRadius = initRadius;
    } // end while attempts

    std::cout << "Loop ended with attempts = " << attempts
              << ", found " << finalCandidates.size() << " patches.\n";

    // Build a finalResult from all the found patches
    PCLResult finalResult;
    finalResult.inlier_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    finalResult.outlier_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    finalResult.downsampled_cloud = cloud;

    // Merge them
    for (auto &patch : candidatePatches) {
        finalResult.inlier_cloud->insert(finalResult.inlier_cloud->end(),
                                         patch.inlier_cloud->begin(),
                                         patch.inlier_cloud->end());
    }
    // Optionally add all original points to outlier
    for (size_t i = 0; i < cloud->size(); i++) {
        finalResult.outlier_cloud->push_back(cloud->points[i]);
    }

    std::cout << "Found " << finalCandidates.size() << " candidate patches.\n";
    std::cout << "Inlier patch= " << finalResult.inlier_cloud->size() << " points.\n";
    std::cout << "Outlier= " << finalResult.outlier_cloud->size() << " points.\n";

    // Return a tuple: (finalResult, vectorOfCandidates)
    return std::make_tuple(finalResult, finalCandidates);
}
//-----------------------------------------------------------------------------
// rankCandidatePatches: sorts all your collected patches by descending score
//-----------------------------------------------------------------------------
inline std::vector<SLZDCandidatePoints> rankCandidatePatches(
    std::vector<SLZDCandidatePoints>& candidatePoints,
    PCLResult& result)
{
    // Sort candidate patches by descending score
    std::sort(candidatePoints.begin(), candidatePoints.end(),
              [](const SLZDCandidatePoints &a, const SLZDCandidatePoints &b) {
                  return a.score > b.score;
              });

    std::cout << "Ranked Candidates (sorted by score):\n";
    for (size_t i = 0; i < candidatePoints.size(); ++i) {
        const auto &cand = candidatePoints[i];
        std::cout << "Rank " << i + 1 << " - Score: " << cand.score << "\n";
        std::cout << "  Data Confidence: " << cand.dataConfidence << "\n";
        std::cout << "  Relief: " << cand.relief << "\n";
        std::cout << "  Roughness: " << cand.roughness << "\n";
        std::cout << "--------------------------\n";
    }
    return candidatePoints;
}


//-----------------------------------------------------------------------------
// visualizeRankedCandidatePatches: displays a rank label for each patch
//-----------------------------------------------------------------------------
inline void visualizeRankedCandidatePatches(const std::vector<SLZDCandidatePoints>& candidatePoints, 
                                            const PCLResult &result)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer(result.pcl_method + " PCL RESULT"));
    viewer->setBackgroundColor(1.0, 1.0, 1.0);

    if (result.outlier_cloud && !result.outlier_cloud->empty()) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> outlierColorHandler(
            result.outlier_cloud, 255, 0, 0);
        viewer->addPointCloud<pcl::PointXYZI>(result.outlier_cloud, outlierColorHandler, "outlier_cloud");
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "outlier_cloud");
    }
    if (result.inlier_cloud && !result.inlier_cloud->empty()) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> inlierColorHandler(
            result.inlier_cloud, 0, 255, 0);
        viewer->addPointCloud<pcl::PointXYZI>(result.inlier_cloud, inlierColorHandler, "inlier_cloud");
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "inlier_cloud");
    }

    std::cout << "Candidate patches size: " << candidatePoints.size() << std::endl;
    for (size_t idx = 0; idx < candidatePoints.size(); ++idx) {
        const auto &candidate = candidatePoints[idx];

        // Add patch surface
        std::stringstream ss;
        ss << "candidate_" << idx << "_surface";
        viewer->addPointCloud<pcl::PointXYZI>(candidate.detectedSurface, ss.str());

        // Use seedPoint as the center for the rank label
        pcl::PointXYZ centerXYZ;
        centerXYZ.x = candidate.seedPoint.x;
        centerXYZ.y = candidate.seedPoint.y;
        centerXYZ.z = candidate.seedPoint.z;

        // Label as "Rank i+1"
        std::stringstream label;
        label << "Rank " << idx + 1;

        viewer->addText3D<pcl::PointXYZ>(
            label.str(), centerXYZ, 2.0, 0.0, 0.0, 0.0, "label_" + std::to_string(idx));
    }

    viewer->resetCamera();
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

#endif
