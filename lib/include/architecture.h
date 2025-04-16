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
#include <pcl/filters/conditional_removal.h>
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
#include <vtkCamera.h>

//-----------------------------------------------------------------------------
// isPatchCollisionFreeVertically: checks if there's an obstacle above the accepted relief margin at ground level of the patch
//-----------------------------------------------------------------------------
// inline bool isPatchCollisionFreeVertically(
//     const pcl::PointCloud<PointPcl>::Ptr &input_cloud,
//     double relief_threshold        //this value is the max acceptable relief (difference between the lowest and highest point in the patch)
// )
// {   // Check if cloud is empty
//     if (input_cloud->empty()) {
//         return false; // No points, no collision
//     }
//     // Initialize with extreme values
//     auto z_min = std::numeric_limits<float>::max();
//     auto z_max = std::numeric_limits<float>::lowest();
//     for (const auto& point : input_cloud->points) {
//         z_min = std::min(z_min, point.z);
//         z_max = std::max(z_max, point.z);
//     }
//     // Create a condition: z > relief_threshold AND z < z_max
//     pcl::ConditionAnd<PointPcl>::Ptr condition(new pcl::ConditionAnd<PointPcl>());

//     // Add condition for z > relief_threshold
//     condition->addComparison(pcl::FieldComparison<PointPcl>::ConstPtr(
//         new pcl::FieldComparison<PointPcl>("z", pcl::ComparisonOps::GT, relief_threshold)));

//     // Add condition for z < z_max
//     condition->addComparison(pcl::FieldComparison<PointPcl>::ConstPtr(
//         new pcl::FieldComparison<PointPcl>("z", pcl::ComparisonOps::LT, z_max)));

//     // Create the ConditionalRemoval filter
//     pcl::ConditionalRemoval<PointPcl> cond_removal;
//     cond_removal.setCondition(condition);
//     cond_removal.setInputCloud(input_cloud);

//     // Apply filter to find points in the range relief_threshold < z < z_max
//     pcl::PointCloud<PointPcl> filtered_cloud;
//     cond_removal.filter(filtered_cloud);

//     // Return true if any points are found (collision), false otherwise
//     return !filtered_cloud.empty();
    

// }
inline bool isPatchCollisionFreeVertically(
    const pcl::PointCloud<PointPcl>::Ptr& input_cloud,
    double relief_threshold)
{
    if (!input_cloud || input_cloud->empty())
    {
        return false; // Cannot be collision-free if empty
    }

    auto z_min = std::numeric_limits<float>::max();
    auto z_max = std::numeric_limits<float>::lowest();
    for (const auto& point : input_cloud->points)
    {
        z_min = std::min(z_min, point.z);
        z_max = std::max(z_max, point.z);
    }

    pcl::ConditionAnd<PointPcl>::Ptr condition(new pcl::ConditionAnd<PointPcl>());
    condition->addComparison(pcl::FieldComparison<PointPcl>::ConstPtr(
        new pcl::FieldComparison<PointPcl>("z", pcl::ComparisonOps::GT, z_min + relief_threshold)));
    condition->addComparison(pcl::FieldComparison<PointPcl>::ConstPtr(
        new pcl::FieldComparison<PointPcl>("z", pcl::ComparisonOps::LT, z_max)));

    pcl::ConditionalRemoval<PointPcl> cond_removal;
    cond_removal.setCondition(condition);
    cond_removal.setInputCloud(input_cloud);

    pcl::PointCloud<PointPcl> filtered_cloud;
    cond_removal.filter(filtered_cloud);

    return filtered_cloud.empty(); // True if no points above relief_threshold
}
//----------------------------------------------------------------------------------------
// kdtreeNeighbourhoodPCAFilterOMP: returns multiple collision-free patches(OMP PARELLIZATION)
//----------------------------------------------------------------------------------------
/**
 * @brief Identifies multiple collision-free landing zone candidate patches in a point cloud using a KdTree and PCA-based filtering.
 *
 * This function uses parallel processing (OMP) to efficiently search for flat, circular patches in the input point cloud
 * that meet specific criteria such as angle threshold, relief threshold, and collision-free requirements.
 *
 * @param input_cloud The input point cloud containing 3D points (pcl::PointXYZI format).
 * @param initRadius The initial radius for the neighborhood search.
 * @param k The number of nearest neighbors to consider for PCA-based analysis.
 * @param angleThreshold The maximum allowable angle deviation for a patch to be considered flat.
 * @param relief_threshold The maximum acceptable relief (difference between the lowest and highest point in the patch).
 * @param maxlandingZones The maximum number of landing zone candidates to identify.
 * @param maxAttempts The maximum number of attempts to find landing zone candidates.
 * @return A vector of LandingZoneCandidatePoint representing the identified landing zone candidates.
 */
// inline std::vector<LandingZoneCandidatePoint> kdtreeNeighbourhoodPCAFilterOMP(
//     const PointCloudPcl& input_cloud,
//     double initRadius,
//     int k,
//     float angleThreshold,
//     float relief_threshold,
//     int maxlandingZones,
//     int maxAttempts)
// {
//     // srand(static_cast<unsigned int>(time(0)));
//     srand(static_cast<unsigned int>(42));


//     std::vector<LandingZoneCandidatePoint> finalCandidates;
//     std::cout << "[kdtreeNeighbourhoodPCAFilterOMP] Loading point cloud..." << std::endl;

//     if (!input_cloud || input_cloud->empty()) {
//         std::cerr << "[kdtreeNeighbourhoodPCAFilterOMP] Error: Loaded point cloud is empty!\n";
//         return finalCandidates;
//     }

//     double minX = std::numeric_limits<double>::max();
//     double maxX = -std::numeric_limits<double>::max();
//     double minY = std::numeric_limits<double>::max();
//     double maxY = -std::numeric_limits<double>::max();
//     double minZ = std::numeric_limits<double>::max();
//     double maxZ = -std::numeric_limits<double>::max();

//     for (const auto &pt : input_cloud->points) {
//         if (pt.x < minX) minX = pt.x;
//         if (pt.x > maxX) maxX = pt.x;
//         if (pt.y < minY) minY = pt.y;
//         if (pt.y > maxY) maxY = pt.y;
//         if (pt.z < minZ) minZ = pt.z;
//         if (pt.z > maxZ) maxZ = pt.z;
//     }

//     std::cout << "[kdtreeNeighbourhoodPCAFilterOMP] Cloud boundaries: "
//               << "x=[" << minX << ", " << maxX << "], "
//               << "y=[" << minY << ", " << maxY << "], "
//               << "z=[" << minZ << ", " << maxZ << "]\n";


//     // Build an KdTree
//     pcl::KdTreeFLANN<PointPcl> kdtree;
//     kdtree.setInputCloud(input_cloud);


//     std::cout << "[kdtreeNeighbourhoodPCAFilterOMP] initRadius=" << initRadius
//               << ", k=" << k
//               << ", angleThreshold=" << angleThreshold
//               << ", maxlandingZones=" << maxlandingZones
//               << ", maxAttempts=" << maxAttempts << "\n\n";
//     omp_lock_t stop_lock;
//     omp_init_lock(&stop_lock); // Initialize the lock

//     #pragma omp parallel for shared(finalCandidates, input_cloud, \
//                                     kdtree, minX, maxX, minY, maxY, stop_lock)
//     for (int attempt = 0; attempt < maxAttempts; ++attempt)
//     {
//         double currentRadius = initRadius;
//         double radiusIncrement = 0.5;
//         double margin = 0.10;   // setting it to 10cm for now, this value is the max acceptable relief (difference between the lowest and highest point in the patch)
//         bool foundFlat = false; // set to true if at least one solution is found
//         processingResult bestFlatPatch;

//         int randomIndex = rand() % input_cloud->size();
//         PointPcl searchPoint = input_cloud->points[randomIndex];

//         #pragma omp critical
//         {
//             std::cout << "[Thread " << omp_get_thread_num()
//                       << "] --> Attempt #" << attempt
//                       << "  Seed=(" << searchPoint.x << ", "
//                       << searchPoint.y << ", "
//                       << searchPoint.z << ")\n";
//         }

//         if (searchPoint.x < (minX + initRadius) || searchPoint.x > (maxX - initRadius) ||
//             searchPoint.y < (minY + initRadius) || searchPoint.y > (maxY - initRadius))
//         {
//             #pragma omp critical
//             {
//                 std::cout << "[Thread " << omp_get_thread_num()
//                           << "] Attempt " << attempt
//                           << ": seed near XY edge => skip.\n";
//             }
//             continue;
//         }

//         while (true) {
//             std::vector<int> idx;
//             std::vector<float> dist;
//             int found = kdtree.radiusSearch(searchPoint, currentRadius, idx, dist);

//             #pragma omp critical
//             {
//                 std::cout << "[Thread " << omp_get_thread_num()
//                           << "] Attempt " << attempt
//                           << "  radius=" << currentRadius
//                           << " => " << found << " neighbors.\n";
//             }
//             std::cout << "[INFO] here__r1: " <<std::endl;

//             if (found < 5) {
//                 break;
//             }
//             std::cout << "[INFO] here__r2: " <<std::endl;

//             auto patch = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
//             patch->reserve(found);
//             for (int i = 0; i < found; i++) {
//                 patch->points.push_back(input_cloud->points[idx[i]]);
//             }
//             std::cout << "[INFO] here__r3: " <<std::endl;

//             processingResult pcaResult = PrincipleComponentAnalysis(patch, angleThreshold, k);
//             std::cout << "[INFO] here__r3.1: " <<std::endl;
//             auto pacaResult_inlier_cloud = std::get<PointCloudPcl>(pcaResult.inlier_cloud);
//             auto pacaResult_outlier_cloud = std::get<PointCloudPcl>(pcaResult.outlier_cloud);
//             std::cout << "[INFO] here__r3.2: " <<std::endl;
//             auto pcaResult_plane_coefficients = std::get<pcl::ModelCoefficients::Ptr>(pcaResult.plane_coefficients);
//             std::cout << "[INFO] here__r4: " <<std::endl;

//             if (pacaResult_outlier_cloud->empty()) {
//                 foundFlat = true;
//                 currentRadius += radiusIncrement;
//                 continue;
//                 }
//             else {
//                 if (foundFlat){
//                     std::cout << "[INFO] here__r5: " <<std::endl;

//                 auto finalSuccessfulRadius = currentRadius;
//                 bool collisionFree = isPatchCollisionFreeVertically(
//                     input_cloud, relief_threshold);
    
//                     if (collisionFree) {
//                         LandingZoneCandidatePoint candidate;
//                         candidate.center.x = searchPoint.x;
//                         candidate.center.y = searchPoint.y;
//                         candidate.center.z = searchPoint.z;
//                         candidate.plane_coefficients = pcaResult.plane_coefficients;
        
//                         candidate.dataConfidence = calculateDataConfidence(pcaResult);
//                         candidate.relief = calculateRelief(pcaResult);
//                         candidate.roughness = calculateRoughness(pcaResult);
//                         candidate.patchRadius = finalSuccessfulRadius;
//                         candidate.score = candidate.dataConfidence
//                                         + candidate.patchRadius
//                                         - candidate.relief
//                                         - candidate.roughness;
//                         candidate.circular_patch = pacaResult_inlier_cloud;
        
//                         #pragma omp critical
//                         {
//                             finalCandidates.push_back(candidate);
        
//                             std::cout << "[Thread " << omp_get_thread_num()
//                                     << "] Attempt " << attempt
//                                     << " => Patch is collision-free, appended!\n";
//                         }
//                     }
//                     else {
//                         #pragma omp critical
//                         {
//                             std::cout << "[Thread " << omp_get_thread_num()
//                                     << "] Attempt " << attempt
//                                     << " => Patch has obstacle => skip.\n";
//                         }
//                     }
//                 }
//                 else {
//                     #pragma omp critical
//                     {
//                         std::cout << "[Thread " << omp_get_thread_num()
//                                 << "] Attempt " << attempt
//                                 << " => No suitable patch from this seed.\n";
//                     }
//                     }
//             }
//             break;
//         }
//     }

//     std::cout << "\n[kdtreeNeighbourhoodPCAFilterOMP] Finished parallel loop.\n"
//               << "  Found " << finalCandidates.size() << " patches.\n";

//     // Cleanup the lock
//     omp_destroy_lock(&stop_lock);
//     std::cout << "[INFO] here__rr: " <<std::endl;


//     return finalCandidates;
// }
inline std::vector<LandingZoneCandidatePoint> kdtreeNeighbourhoodPCAFilterOMP(
    const PointCloudPcl& input_cloud,
    double initRadius,
    int k,
    float angleThreshold,
    float relief_threshold,
    int maxlandingZones,
    int maxAttempts)
{
    std::vector<LandingZoneCandidatePoint> finalCandidates;
    #pragma omp critical
    {
        std::cout << "[kdtreeNeighbourhoodPCAFilterOMP] Loading point cloud...\n";
    }

    if (!input_cloud || input_cloud->empty())
    {
        #pragma omp critical
        {
            std::cerr << "[kdtreeNeighbourhoodPCAFilterOMP] Error: Loaded point cloud is empty!\n";
        }
        return finalCandidates;
    }

    double minX = std::numeric_limits<double>::max();
    double maxX = -std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxY = -std::numeric_limits<double>::max();
    double minZ = std::numeric_limits<double>::max();
    double maxZ = -std::numeric_limits<double>::max();

    for (const auto& pt : input_cloud->points)
    {
        minX = std::min(minX, static_cast<double>(pt.x));
        maxX = std::max(maxX, static_cast<double>(pt.x));
        minY = std::min(minY, static_cast<double>(pt.y));
        maxY = std::max(maxY, static_cast<double>(pt.y));
        minZ = std::min(minZ, static_cast<double>(pt.z));
        maxZ = std::max(maxZ, static_cast<double>(pt.z));
    }

    #pragma omp critical
    {
        std::cout << "[kdtreeNeighbourhoodPCAFilterOMP] Cloud boundaries: "
                  << "x=[" << minX << ", " << maxX << "], "
                  << "y=[" << minY << ", " << maxY << "], "
                  << "z=[" << minZ << ", " << maxZ << "]\n";
        std::cout << "[kdtreeNeighbourhoodPCAFilterOMP] initRadius=" << initRadius
                  << ", k=" << k
                  << ", angleThreshold=" << angleThreshold
                  << ", maxlandingZones=" << maxlandingZones
                  << ", maxAttempts=" << maxAttempts << "\n\n";
    }

    pcl::KdTreeFLANN<PointPcl> kdtree;
    kdtree.setInputCloud(input_cloud);

    bool cancel_flag = false;

    #pragma omp parallel for shared(finalCandidates, input_cloud, kdtree, minX, maxX, minY, maxY, cancel_flag)
    for (int attempt = 0; attempt < maxAttempts; ++attempt)
    {
        #pragma omp cancellation point for
        if (cancel_flag)
        {
            continue; // Skip if cancellation signaled
        }

        double currentRadius = initRadius;
        double radiusIncrement = 0.5;
        bool foundFlat = false;
        processingResult bestFlatPatch;

        // Thread-local random number generator
        thread_local std::mt19937 rng(std::random_device{}());
        int randomIndex = std::uniform_int_distribution<>(0, input_cloud->size() - 1)(rng);
        PointPcl searchPoint = input_cloud->points[randomIndex];

        #pragma omp critical
        {
            std::cout << "[Thread " << omp_get_thread_num()
                      << "] --> Attempt #" << attempt
                      << "  Seed=(" << searchPoint.x << ", "
                      << searchPoint.y << ", "
                      << searchPoint.z << ")\n";
        }

        if (searchPoint.x < (minX + initRadius) || searchPoint.x > (maxX - initRadius) ||
            searchPoint.y < (minY + initRadius) || searchPoint.y > (maxY - initRadius))
        {
            #pragma omp critical
            {
                std::cout << "[Thread " << omp_get_thread_num()
                          << "] Attempt " << attempt
                          << ": seed near XY edge => skip.\n";
            }
            continue;
        }

        while (true)
        {
            std::vector<int> idx;
            std::vector<float> dist;
            int found = kdtree.radiusSearch(searchPoint, currentRadius, idx, dist);

            #pragma omp critical
            {
                std::cout << "[Thread " << omp_get_thread_num()
                          << "] Attempt " << attempt
                          << "  radius=" << currentRadius
                          << " => " << found << " neighbors.\n";
            }

            if (found < k)
            {
                #pragma omp critical
                {
                    std::cout << "[Thread " << omp_get_thread_num()
                              << "] Attempt " << attempt
                              << ": too few neighbors (" << found << ") => skip.\n";
                }
                break;
            }

            auto patch = std::make_shared<pcl::PointCloud<PointPcl>>();
            patch->reserve(found);
            for (int i = 0; i < found; ++i)
            {
                patch->points.push_back(input_cloud->points[idx[i]]);
            }
            patch->width = patch->points.size();
            patch->height = 1;
            patch->is_dense = true;

            processingResult pcaResult = PrincipleComponentAnalysis(patch, angleThreshold, k);

            if (!std::holds_alternative<PointCloudPcl>(pcaResult.inlier_cloud) ||
                !std::holds_alternative<PointCloudPcl>(pcaResult.outlier_cloud))
            {
                #pragma omp critical
                {
                    std::cerr << "[Thread " << omp_get_thread_num()
                              << "] Attempt " << attempt
                              << ": PCA returned invalid cloud type\n";
                }
                break;
            }

            auto pcaResult_inlier_cloud = std::get<PointCloudPcl>(pcaResult.inlier_cloud);
            auto pcaResult_outlier_cloud = std::get<PointCloudPcl>(pcaResult.outlier_cloud);

            if (!pcaResult_inlier_cloud || !pcaResult_outlier_cloud)
            {
                #pragma omp critical
                {
                    std::cerr << "[Thread " << omp_get_thread_num()
                              << "] Attempt " << attempt
                              << ": PCA returned null clouds\n";
                }
                break;
            }
            if (!std::holds_alternative<pcl::ModelCoefficients::Ptr>(pcaResult.plane_coefficients))
            {
                #pragma omp critical
                {
                    std::cerr << "[Thread " << omp_get_thread_num()
                              << "] Attempt " << attempt
                              << ": PCA returned invalid plane coefficients\n";
                }
                break;
            }

            auto pcaResult_plane_coefficients = std::get<pcl::ModelCoefficients::Ptr>(pcaResult.plane_coefficients);
            if (!pcaResult_plane_coefficients || pcaResult_plane_coefficients->values.size() < 4)
            {
                #pragma omp critical
                {
                    std::cerr << "[Thread " << omp_get_thread_num()
                              << "] Attempt " << attempt
                              << ": PCA returned null or invalid plane coefficients\n";
                }
                break;
            }
            if (pcaResult_outlier_cloud->size() <= pcaResult_inlier_cloud->size() * 0.05) // Allow 5% outliers
            {
                foundFlat = true;
                bestFlatPatch = pcaResult;
                currentRadius += radiusIncrement;
                continue;
            }
            else if (foundFlat)
            {

                auto finalSuccessfulRadius = currentRadius - radiusIncrement;
                bool collisionFree = !isPatchCollisionFreeVertically(pcaResult_inlier_cloud, relief_threshold);

                if (collisionFree)
                {
                    LandingZoneCandidatePoint candidate;
                    candidate.center.x = searchPoint.x;
                    candidate.center.y = searchPoint.y;
                    candidate.center.z = searchPoint.z;
                    candidate.plane_coefficients = pcaResult.plane_coefficients;
                    candidate.dataConfidence = calculateDataConfidence(pcaResult);
                    candidate.relief = calculateRelief(pcaResult);
                    candidate.roughness = calculateRoughness(pcaResult);
                    candidate.patchRadius = finalSuccessfulRadius;
                    candidate.score = candidate.dataConfidence + candidate.patchRadius - candidate.relief - candidate.roughness;
                    candidate.circular_patch = pcaResult_inlier_cloud;

                    #pragma omp critical
                    {
                        finalCandidates.push_back(candidate);
                        std::cout << "[Thread " << omp_get_thread_num()
                                  << "] Attempt " << attempt
                                  << " => Patch is collision-free, appended!\n";
                    }

                    // Check if max landing zones reached
                    if (finalCandidates.size() >= static_cast<size_t>(maxlandingZones))
                    {
                        #pragma omp critical
                        {
                            std::cout << "[Thread " << omp_get_thread_num()
                                      << "] Reached max landing zones (" << maxlandingZones << "), signaling cancellation.\n";
                        }
                        cancel_flag = true;
                        #pragma omp cancel for
                    }
                }
                else
                {
                    #pragma omp critical
                    {
                        std::cout << "[Thread " << omp_get_thread_num()
                                  << "] Attempt " << attempt
                                  << " => Patch has obstacle => skip.\n";
                    }
                }
            }
            else
            {
                #pragma omp critical
                {
                    std::cout << "[Thread " << omp_get_thread_num()
                              << "] Attempt " << attempt
                              << " => No suitable patch from this seed.\n";
                }
            }
            break;
        }
    }

    #pragma omp critical
    {
        std::cout << "\n[kdtreeNeighbourhoodPCAFilterOMP] Finished parallel loop.\n"
                  << "  Found " << finalCandidates.size() << " patches.\n";
    }

    return finalCandidates;
}

//-----------------------------------------------------------------------------
// rankCandidates: sorts all your collected patches by descending score
//-----------------------------------------------------------------------------
inline std::vector<LandingZoneCandidatePoint> rankCandidatePatches(
    std::vector<LandingZoneCandidatePoint>& candidatePoints)
{
    // Sort candidate patches by descending score
    std::sort(candidatePoints.begin(), candidatePoints.end(),
              [](const LandingZoneCandidatePoint &a, const LandingZoneCandidatePoint &b) {
                  return a.score > b.score;
              });

    std::cout << "Ranked Candidates (sorted by score):\n";
    for (size_t i = 0; i < candidatePoints.size(); ++i) {
        const auto &cand = candidatePoints[i];
        std::cout << "Rank " << i + 1 << " - Score: " << cand.score << "\n";
        std::cout << "  Data Confidence: " << cand.dataConfidence << "\n";
        std::cout << "  Relief: " << cand.relief << "\n";
        std::cout << "  Roughness: " << cand.roughness << "\n";
        std::cout << "  Patch Radius: " << cand.patchRadius << "\n";
        std::cout << "--------------------------\n";
    }
    return candidatePoints;
}


//-----------------------------------------------------------------------------
// visualizeRankedCandidates: displays a rank label for each patch
//-----------------------------------------------------------------------------

inline void visualizeRankedCandidates(const std::vector<LandingZoneCandidatePoint>& candidatePoints, 
    const processingResult& result, float textSize)
{
// Initialize viewer
pcl::visualization::PCLVisualizer::Ptr viewer(
new pcl::visualization::PCLVisualizer("Ranked Candidates Visualization"));
viewer->setBackgroundColor(1.0, 1.0, 1.0);

// Fixed color for visualization
pcl::RGB fixedColor = {255, 0, 0}; // Red color

// Visualize inliers and outliers, handling PointCloud variant
if (std::holds_alternative<PointCloudPcl>(result.inlier_cloud)) {
auto result_inlier_cloud = std::get<PointCloudPcl>(result.inlier_cloud);
if (result_inlier_cloud && !result_inlier_cloud->empty()) {
pcl::visualization::PointCloudColorHandlerCustom<PointPcl> inlierColorHandler(
result_inlier_cloud, 0, 255, 0);
viewer->addPointCloud<PointPcl>(result_inlier_cloud, inlierColorHandler, "inlier_cloud");
viewer->setPointCloudRenderingProperties(
pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "inlier_cloud");
}
}
if (std::holds_alternative<PointCloudPcl>(result.outlier_cloud)) {
auto result_outlier_cloud = std::get<PointCloudPcl>(result.outlier_cloud);
if (result_outlier_cloud && !result_outlier_cloud->empty()) {
pcl::visualization::PointCloudColorHandlerCustom<PointPcl> outlierColorHandler(
result_outlier_cloud, 255, 0, 0);
viewer->addPointCloud<PointPcl>(result_outlier_cloud, outlierColorHandler, "outlier_cloud");
viewer->setPointCloudRenderingProperties(
pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "outlier_cloud");
}
}

std::cout << "Candidate patches size: " << candidatePoints.size() << std::endl;

// Prepare data in a thread-safe way
std::vector<std::tuple<int, PointPcl, PointCloudPcl, float>> patchData;

// Parallelize the data processing part
#pragma omp parallel for default(none) shared(candidatePoints, patchData)
for (size_t idx = 0; idx < candidatePoints.size(); ++idx) {
const auto& candidate = candidatePoints[idx];

// Calculate the center point of the patch
PointPcl centerXYZ;
centerXYZ.x = candidate.center.x;
centerXYZ.y = candidate.center.y;
centerXYZ.z = candidate.center.z;

// Extract PointCloudPcl from candidate.circular_patch
PointCloudPcl circular_patch;
if (std::holds_alternative<PointCloudPcl>(candidate.circular_patch)) {
circular_patch = std::get<PointCloudPcl>(candidate.circular_patch);
} else {
// Convert Open3D to PCL if necessary
auto open3d_patch = std::get<PointCloudOpen3D>(candidate.circular_patch);
circular_patch = std::make_shared<pcl::PointCloud<PointPcl>>();
circular_patch->points.reserve(open3d_patch->points_.size());
for (const auto& point : open3d_patch->points_) {
PointPcl p;
p.x = point(0);
p.y = point(1);
p.z = point(2);
p.intensity = 1.0; // Adjust as needed
circular_patch->push_back(p);
}
circular_patch->width = circular_patch->points.size();
circular_patch->height = 1;
circular_patch->is_dense = true;
}

// Store the required data for visualization
#pragma omp critical
patchData.push_back(std::make_tuple(static_cast<int>(idx), centerXYZ, circular_patch, static_cast<float>(candidate.patchRadius)));
}

// Visualize the patches
for (const auto& data : patchData) {
int idx;
PointPcl centerXYZ;
PointCloudPcl circular_patch;
float radius;
std::tie(idx, centerXYZ, circular_patch, radius) = data;

// Visualize the patch surface
std::stringstream ss;
ss << "candidate_" << idx << "_surface";
viewer->addPointCloud<PointPcl>(circular_patch, ss.str());

// Add center point as a sphere
std::stringstream centerName;
centerName << "center_" << idx;
viewer->addSphere(centerXYZ, 0.1, fixedColor.r / 255.0, fixedColor.g / 255.0, fixedColor.b / 255.0, centerName.str());

// Label as "Rank i+1"
std::stringstream label;
label << "Rank " << idx + 1;
viewer->addText3D(label.str(), centerXYZ, textSize, fixedColor.r / 255.0, fixedColor.g / 255.0, fixedColor.b / 255.0, "label_" + std::to_string(idx));

// Generate and visualize the boundary (circumference of the patch)
int num_points = 10000; // Number of points to form the circle
PointCloudPcl circle = std::make_shared<pcl::PointCloud<PointPcl>>();

// Generate points for the circumference
for (int i = 0; i < num_points; ++i) {
float angle = 2 * M_PI * i / num_points;
PointPcl pt;
pt.x = centerXYZ.x + radius * cos(angle);
pt.y = centerXYZ.y + radius * sin(angle);
pt.z = centerXYZ.z;
pt.intensity = 0.0; // Set intensity as needed
circle->points.push_back(pt);
}
circle->width = circle->points.size();
circle->height = 1;
circle->is_dense = true;

// Visualize the boundary (circle)
std::stringstream circleName;
circleName << "circle_" << idx;
pcl::visualization::PointCloudColorHandlerCustom<PointPcl> circleColorHandler(
circle, fixedColor.r, fixedColor.g, fixedColor.b);
viewer->addPointCloud<PointPcl>(circle, circleColorHandler, circleName.str());
}

viewer->resetCamera();
while (!viewer->wasStopped()) {
viewer->spinOnce(100);
std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
}

inline void visualizePCLWithRankedCandidates(
    const processingResult& result,
    const std::vector<LandingZoneCandidatePoint>& candidatePoints,
    const std::string& viz_inlier_or_outlier_or_both = "both",
    float textSize = 0.5f)
{
    std::cout << "[INFO] visualizePCLWithRankedCandidates: Starting visualization\n";

    // Create a single visualizer object
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("Point Cloud with Ranked Candidates: " + 
                                             result.processing_method + " (" + result.hazardMetric_type + ")"));
    viewer->setBackgroundColor(1.0, 1.0, 1.0);
    std::cout << "[INFO] visualizePCLWithRankedCandidates: Viewer initialized\n";

    // Initialize PCL clouds
    std::shared_ptr<pcl::PointCloud<PointPcl>> pcl_inlier_cloud = nullptr;
    std::shared_ptr<pcl::PointCloud<PointPcl>> pcl_outlier_cloud = nullptr;

    // Variables for bounding box
    Eigen::Vector4f min_pt(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 
                           std::numeric_limits<float>::max(), 1.0f);
    Eigen::Vector4f max_pt(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), 
                           std::numeric_limits<float>::lowest(), 1.0f);
    Eigen::Vector4f centroid(0.0f, 0.0f, 0.0f, 1.0f);

    // Handle inlier cloud
    if (std::holds_alternative<PointCloudPcl>(result.inlier_cloud)) {
        pcl_inlier_cloud = std::get<PointCloudPcl>(result.inlier_cloud);
        std::cout << "[INFO] visualizePCLWithRankedCandidates: Inlier cloud is PCL, size: " 
                  << (pcl_inlier_cloud ? pcl_inlier_cloud->size() : 0) << "\n";
        if (pcl_inlier_cloud && !pcl_inlier_cloud->empty()) {
            pcl::PointXYZI min, max;
            pcl::getMinMax3D(*pcl_inlier_cloud, min, max);
            min_pt = Eigen::Vector4f(min.x, min.y, min.z, 1.0f);
            max_pt = Eigen::Vector4f(max.x, max.y, max.z, 1.0f);
            pcl::compute3DCentroid(*pcl_inlier_cloud, centroid);
            std::cout << "[INFO] visualizePCLWithRankedCandidates: Inlier bounds: min=(" 
                      << min.x << ", " << min.y << ", " << min.z << "), max=(" 
                      << max.x << ", " << max.y << ", " << max.z << "), centroid=(" 
                      << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << ")\n";
        }
    } else if (std::holds_alternative<PointCloudOpen3D>(result.inlier_cloud)) {
        std::cout << "[INFO] visualizePCLWithRankedCandidates: Converting Open3D inlier cloud to PCL\n";
        auto open3d_cloud = std::get<PointCloudOpen3D>(result.inlier_cloud);
        pcl_inlier_cloud = std::make_shared<pcl::PointCloud<PointPcl>>();
        if (open3d_cloud && !open3d_cloud->points_.empty()) {
            pcl_inlier_cloud->points.reserve(open3d_cloud->points_.size());
            double min_x = std::numeric_limits<double>::max(), min_y = min_x, min_z = min_x;
            double max_x = std::numeric_limits<double>::lowest(), max_y = max_x, max_z = max_x;
            double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
            size_t count = 0;
            for (const auto& point : open3d_cloud->points_) {
                PointPcl p;
                p.x = point(0);
                p.y = point(1);
                p.z = point(2);
                p.intensity = 1.0;
                pcl_inlier_cloud->push_back(p);
                min_x = std::min(min_x, point(0));
                min_y = std::min(min_y, point(1));
                min_z = std::min(min_z, point(2));
                max_x = std::max(max_x, point(0));
                max_y = std::max(max_y, point(1));
                max_z = std::max(max_z, point(2));
                sum_x += point(0);
                sum_y += point(1);
                sum_z += point(2);
                count++;
            }
            pcl_inlier_cloud->width = pcl_inlier_cloud->size();
            pcl_inlier_cloud->height = 1;
            pcl_inlier_cloud->is_dense = true;
            min_pt = Eigen::Vector4f(min_x, min_y, min_z, 1.0f);
            max_pt = Eigen::Vector4f(max_x, max_y, max_z, 1.0f);
            centroid = Eigen::Vector4f(sum_x / count, sum_y / count, sum_z / count, 1.0f);
            std::cout << "[INFO] visualizePCLWithRankedCandidates: Inlier bounds: min=(" 
                      << min_x << ", " << min_y << ", " << min_z << "), max=(" 
                      << max_x << ", " << max_y << ", " << max_z << "), centroid=(" 
                      << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << ")\n";
        }
        std::cout << "[INFO] visualizePCLWithRankedCandidates: Inlier cloud (converted), size: " 
                  << (pcl_inlier_cloud ? pcl_inlier_cloud->size() : 0) << "\n";
    } else {
        std::cerr << "[ERROR] visualizePCLWithRankedCandidates: Unknown variant type for inlier_cloud\n";
    }

    // Handle outlier cloud
    if (std::holds_alternative<PointCloudPcl>(result.outlier_cloud)) {
        pcl_outlier_cloud = std::get<PointCloudPcl>(result.outlier_cloud);
        std::cout << "[INFO] visualizePCLWithRankedCandidates: Outlier cloud is PCL, size: " 
                  << (pcl_outlier_cloud ? pcl_outlier_cloud->size() : 0) <<"\n";
        if (pcl_outlier_cloud && !pcl_outlier_cloud->empty()) {
            pcl::PointXYZI min, max;
            pcl::getMinMax3D(*pcl_outlier_cloud, min, max);
            min_pt = Eigen::Vector4f(std::min(min_pt[0], min.x), std::min(min_pt[1], min.y), 
                                     std::min(min_pt[2], min.z), 1.0f);
            max_pt = Eigen::Vector4f(std::max(max_pt[0], max.x), std::max(max_pt[1], max.y), 
                                     std::max(max_pt[2], max.z), 1.0f);
            std::cout << "[INFO] visualizePCLWithRankedCandidates: Outlier bounds: min=(" 
                      << min.x << ", " << min.y << ", " << min.z << "), max=(" 
                      << max.x << ", " << max.y << ", " << max.z << ")\n";
        }
    } else if (std::holds_alternative<PointCloudOpen3D>(result.outlier_cloud)) {
        std::cout << "[INFO] visualizePCLWithRankedCandidates: Converting Open3D outlier cloud to PCL\n";
        auto open3d_cloud = std::get<PointCloudOpen3D>(result.outlier_cloud);
        pcl_outlier_cloud = std::make_shared<pcl::PointCloud<PointPcl>>();
        if (open3d_cloud && !open3d_cloud->points_.empty()) {
            pcl_outlier_cloud->points.reserve(open3d_cloud->points_.size());
            double min_x = std::numeric_limits<double>::max(), min_y = min_x, min_z = min_x;
            double max_x = std::numeric_limits<double>::lowest(), max_y = max_x, max_z = max_x;
            for (const auto& point : open3d_cloud->points_) {
                PointPcl p;
                p.x = point(0);
                p.y = point(1);
                p.z = point(2);
                p.intensity = 1.0;
                pcl_outlier_cloud->push_back(p);
                min_x = std::min(min_x, point(0));
                min_y = std::min(min_y, point(1));
                min_z = std::min(min_z, point(2));
                max_x = std::max(max_x, point(0));
                max_y = std::max(max_y, point(1));
                max_z = std::max(max_z, point(2));
            }
            pcl_outlier_cloud->width = pcl_outlier_cloud->size();
            pcl_outlier_cloud->height = 1;
            pcl_outlier_cloud->is_dense = true;
            min_pt = Eigen::Vector4f(std::min(min_pt[0], (float)min_x), std::min(min_pt[1], (float)min_y), 
                                     std::min(min_pt[2], (float)min_z), 1.0f);
            max_pt = Eigen::Vector4f(std::max(max_pt[0], (float)max_x), std::max(max_pt[1], (float)max_y), 
                                     std::max(max_pt[2], (float)max_z), 1.0f);
            std::cout << "[INFO] visualizePCLWithRankedCandidates: Outlier bounds: min=(" 
                      << min_x << ", " << min_y << ", " << min_z << "), max=(" 
                      << max_x << ", " << max_y << ", " << max_z << ")\n";
        }
        std::cout << "[INFO] visualizePCLWithRankedCandidates: Outlier cloud (converted), size: " 
                  << (pcl_outlier_cloud ? pcl_outlier_cloud->size() : 0) << "\n";
    } else {
        std::cerr << "[ERROR] visualizePCLWithRankedCandidates: Unknown variant type for outlier_cloud\n";
    }

    // Add inlier cloud (green)
    if (pcl_inlier_cloud && !pcl_inlier_cloud->empty() &&
        (viz_inlier_or_outlier_or_both == "inlier_cloud" || viz_inlier_or_outlier_or_both == "both"))
    {
        pcl::visualization::PointCloudColorHandlerCustom<PointPcl> inlierColorHandler(pcl_inlier_cloud, 0, 255, 0);
        viewer->addPointCloud<PointPcl>(pcl_inlier_cloud, inlierColorHandler, "inlier_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "inlier_cloud");
        std::cout << "[INFO] visualizePCLWithRankedCandidates: Added inlier cloud to viewer\n";
    }

    // Add outlier cloud (red)
    if (pcl_outlier_cloud && !pcl_outlier_cloud->empty() &&
        (viz_inlier_or_outlier_or_both == "outlier_cloud" || viz_inlier_or_outlier_or_both == "both"))
    {
        pcl::visualization::PointCloudColorHandlerCustom<PointPcl> outlierColorHandler(pcl_outlier_cloud, 255, 0, 0);
        viewer->addPointCloud<PointPcl>(pcl_outlier_cloud, outlierColorHandler, "outlier_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "outlier_cloud");
        std::cout << "[INFO] visualizePCLWithRankedCandidates: Added outlier cloud to viewer\n";
    }

    // Visualize ranked candidates
    std::cout << "[INFO] visualizePCLWithRankedCandidates: Candidate patches size: " << candidatePoints.size() << "\n";
    pcl::RGB candidateColor = {255, 165, 0}; // Orange

    for (size_t idx = 0; idx < candidatePoints.size(); ++idx)
    {
        const auto& candidate = candidatePoints[idx];

        // Extract circular patch
        PointCloudPcl circular_patch = nullptr;
        if (std::holds_alternative<PointCloudPcl>(candidate.circular_patch)) {
            circular_patch = std::get<PointCloudPcl>(candidate.circular_patch);
        } else if (std::holds_alternative<PointCloudOpen3D>(candidate.circular_patch)) {
            std::cout << "[INFO] visualizePCLWithRankedCandidates: Converting Open3D circular_patch for candidate " 
                      << idx << " to PCL\n";
            auto open3d_patch = std::get<PointCloudOpen3D>(candidate.circular_patch);
            circular_patch = std::make_shared<pcl::PointCloud<PointPcl>>();
            if (open3d_patch && !open3d_patch->points_.empty()) {
                circular_patch->points.reserve(open3d_patch->points_.size());
                for (const auto& point : open3d_patch->points_) {
                    PointPcl p;
                    p.x = point(0);
                    p.y = point(1);
                    p.z = point(2);
                    p.intensity = 1.0;
                    circular_patch->push_back(p);
                }
                circular_patch->width = circular_patch->size();
                circular_patch->height = 1;
                circular_patch->is_dense = true;
            }
        }

        // Update bounding box with patch
        if (circular_patch && !circular_patch->empty()) {
            pcl::PointXYZI min, max;
            pcl::getMinMax3D(*circular_patch, min, max);
            min_pt = Eigen::Vector4f(std::min(min_pt[0], min.x), std::min(min_pt[1], min.y), 
                                     std::min(min_pt[2], min.z), 1.0f);
            max_pt = Eigen::Vector4f(std::max(max_pt[0], max.x), std::max(max_pt[1], max.y), 
                                     std::max(max_pt[2], max.z), 1.0f);
        }

        // Log candidate info
        std::cout << "[INFO] visualizePCLWithRankedCandidates: Candidate " << idx 
                  << " patch size: " << (circular_patch ? circular_patch->size() : 0) 
                  << ", center: (" << candidate.center.x << ", " << candidate.center.y 
                  << ", " << candidate.center.z << "), radius: " << candidate.patchRadius << "\n";

        // Visualize patch surface (orange)
        if (circular_patch && !circular_patch->empty()) {
            std::stringstream ss;
            ss << "candidate_" << idx << "_surface";
            pcl::visualization::PointCloudColorHandlerCustom<PointPcl> patchColorHandler(circular_patch, candidateColor.r, candidateColor.g, candidateColor.b);
            viewer->addPointCloud<PointPcl>(circular_patch, patchColorHandler, ss.str());
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, ss.str());
        }

        // Add center point as a sphere
        PointPcl centerXYZ = candidate.center;
        min_pt = Eigen::Vector4f(std::min(min_pt[0], centerXYZ.x), std::min(min_pt[1], centerXYZ.y), 
                                 std::min(min_pt[2], centerXYZ.z), 1.0f);
        max_pt = Eigen::Vector4f(std::max(max_pt[0], centerXYZ.x), std::max(max_pt[1], centerXYZ.y), 
                                 std::max(max_pt[2], centerXYZ.z), 1.0f);
        std::stringstream centerName;
        centerName << "center_" << idx;
        viewer->addSphere(centerXYZ, 0.1, candidateColor.r / 255.0, candidateColor.g / 255.0, candidateColor.b / 255.0, centerName.str());

        // Add rank label
        std::stringstream label;
        label << "Rank " << idx + 1;
        viewer->addText3D(label.str(), centerXYZ, textSize, candidateColor.r / 255.0, candidateColor.g / 255.0, candidateColor.b / 255.0, "label_" + std::to_string(idx));

        // Generate and visualize boundary circle
        float radius = candidate.patchRadius;
        PointCloudPcl circle = std::make_shared<pcl::PointCloud<PointPcl>>();
        int num_points = 1000;
        for (int i = 0; i < num_points; ++i) {
            float angle = 2 * M_PI * i / num_points;
            PointPcl pt;
            pt.x = centerXYZ.x + radius * cos(angle);
            pt.y = centerXYZ.y + radius * sin(angle);
            pt.z = centerXYZ.z;
            pt.intensity = 0.0;
            circle->points.push_back(pt);
        }
        circle->width = circle->points.size();
        circle->height = 1;
        circle->is_dense = true;

        std::stringstream circleName;
        circleName << "circle_" << idx;
        pcl::visualization::PointCloudColorHandlerCustom<PointPcl> circleColorHandler(circle, candidateColor.r, candidateColor.g, candidateColor.b);
        viewer->addPointCloud<PointPcl>(circle, circleColorHandler, circleName.str());
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, circleName.str());
    }

    // Log if no points were added
    if (!(pcl_inlier_cloud && !pcl_inlier_cloud->empty()) && 
        !(pcl_outlier_cloud && !pcl_outlier_cloud->empty()) && 
        candidatePoints.empty()) {
        std::cout << "[WARNING] visualizePCLWithRankedCandidates: No points or candidates to display\n";
    }

    // Adjust camera based on bounding box
    if (min_pt[0] <= max_pt[0]) { // Valid bounds
        float scene_size = std::max({max_pt[0] - min_pt[0], max_pt[1] - min_pt[1], max_pt[2] - min_pt[2]});
        float camera_distance = scene_size * 2.0f; // Position camera to encompass scene
        Eigen::Vector4f camera_pos = centroid + Eigen::Vector4f(0.0f, 0.0f, camera_distance, 0.0f);
        viewer->setCameraPosition(
            camera_pos[0], camera_pos[1], camera_pos[2], // Camera position
            centroid[0], centroid[1], centroid[2],       // Look at centroid
            0.0, 1.0, 0.0                               // Up direction
        );
        std::cout << "[INFO] visualizePCLWithRankedCandidates: Camera set to pos=(" 
                  << camera_pos[0] << ", " << camera_pos[1] << ", " << camera_pos[2] 
                  << "), looking at (" << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << ")\n";
    } else {
        std::cout << "[WARNING] visualizePCLWithRankedCandidates: Invalid bounds, using default camera\n";
        viewer->resetCamera();
    }

    // Add coordinate system
    viewer->addCoordinateSystem(1.0);
    std::cout << "[INFO] visualizePCLWithRankedCandidates: Camera and axes set\n";

    // Main loop
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "[INFO] visualizePCLWithRankedCandidates: Visualizer closed\n";
}

#endif
