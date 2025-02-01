#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>
#include <pcl/filters/boost.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/impl/pcl_base.h>
#include <pcl/common/common.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>


struct PointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint16_t ring;
    double timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)
    (float, intensity, intensity)
    (uint16_t, ring, ring)
    (double, timestamp, timestamp)
)

class LOAMOdometry {
public:
    LOAMOdometry() : nh_("~"), map_cloud_(new pcl::PointCloud<PointXYZIRT>()) {
        // Initialize parameters
        nh_.param<int>("num_scans", num_scans_, 16);
        nh_.param<float>("edge_threshold", edge_threshold_, 1.0);
        nh_.param<float>("surf_threshold", surf_threshold_, 0.1);
        nh_.param<float>("map_resolution", map_resolution_, 0.4);
        
        // Initialize ROS components
        cloud_sub_ = nh_.subscribe("/velodyne_points", 10, &LOAMOdometry::cloudCallback, this);
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/loam_odom", 10);
        map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/global_map", 1);
        
        // Initialize map components
        map_tree_.setInputCloud(map_cloud_);
        downsampler_.setLeafSize(map_resolution_, map_resolution_, map_resolution_);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    ros::Publisher odom_pub_, map_pub_;
    
    pcl::PointCloud<PointXYZIRT>::Ptr map_cloud_;
    pcl::KdTreeFLANN<PointXYZIRT> map_tree_;
    pcl::VoxelGrid<PointXYZIRT> downsampler_;
    
    Eigen::Matrix4f pose_ = Eigen::Matrix4f::Identity();
    int num_scans_;
    float edge_threshold_, surf_threshold_, map_resolution_;

    struct FeatureClouds {
        pcl::PointCloud<PointXYZIRT>::Ptr edge;
        pcl::PointCloud<PointXYZIRT>::Ptr surf;
        
        FeatureClouds() : edge(new pcl::PointCloud<PointXYZIRT>), 
                         surf(new pcl::PointCloud<PointXYZIRT>) {}
    };

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        // Convert ROS message to PCL cloud
        pcl::PointCloud<PointXYZIRT>::Ptr cloud(new pcl::PointCloud<PointXYZIRT>);
        pcl::fromROSMsg(*msg, *cloud);
        
        // Process features and estimate odometry
        FeatureClouds features = extractFeatures(cloud);
        if(!features.edge->empty() && !features.surf->empty()) {
            estimateOdometry(features);
            updateMap(features);
            publishOdometry(msg->header.stamp);
        }
    }

    FeatureClouds extractFeatures(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud) {
        FeatureClouds features;
        std::vector<pcl::PointCloud<PointXYZIRT>::Ptr> scan_bins(num_scans_);
        
        // Initialize scan containers
        for(auto& scan : scan_bins) scan.reset(new pcl::PointCloud<PointXYZIRT>);
        
        // Organize points into scans
        for(const auto& pt : *cloud) {
            if(pt.ring < num_scans_) scan_bins[pt.ring]->push_back(pt);
        }

        // Process each scan
        for(auto& scan : scan_bins) {
            if(scan->size() < 10) continue;
            
            std::vector<float> curvatures(scan->size(), 0.0f);
            
            // Calculate curvature for each point
            for(size_t i = 5; i < scan->size()-5; ++i) {
                float diff = 0.0f;
                for(int j = -5; j <= 5; ++j) {
                    if(j != 0) diff += fabs(scan->points[i+j].z - scan->points[i].z);
                }
                curvatures[i] = diff;
            }
            
            // Extract features from this scan
            extractScanFeatures(scan, curvatures, features);
        }
        
        return features;
    }

    void extractScanFeatures(const pcl::PointCloud<PointXYZIRT>::Ptr& scan,
                            const std::vector<float>& curvatures,
                            FeatureClouds& features) {
        std::vector<size_t> edge_indices, surf_indices;
        
        // Classify points based on curvature
        for(size_t i = 0; i < curvatures.size(); ++i) {
            if(curvatures[i] > edge_threshold_) edge_indices.push_back(i);
            else if(curvatures[i] < surf_threshold_) surf_indices.push_back(i);
        }
        
        // Sort and select strongest features
        auto edge_compare = [&](size_t a, size_t b) { return curvatures[a] > curvatures[b]; };
        auto surf_compare = [&](size_t a, size_t b) { return curvatures[a] < curvatures[b]; };
        
        std::sort(edge_indices.begin(), edge_indices.end(), edge_compare);
        std::sort(surf_indices.begin(), surf_indices.end(), surf_compare);
        
        // Select top 20% of each feature type
        size_t edge_count = edge_indices.size() / 5;
        size_t surf_count = surf_indices.size() / 5;
        
        for(size_t i = 0; i < edge_count && i < edge_indices.size(); ++i)
            features.edge->push_back(scan->points[edge_indices[i]]);
        
        for(size_t i = 0; i < surf_count && i < surf_indices.size(); ++i)
            features.surf->push_back(scan->points[surf_indices[i]]);
    }

    struct EdgeCostFunctor {
        EdgeCostFunctor(const Eigen::Vector3f& current,
                       const Eigen::Vector3f& p1,
                       const Eigen::Vector3f& p2)
            : current_point(current), map_point1(p1), map_point2(p2) {}
        
        template <typename T>
        bool operator()(const T* const pose, T* residual) const {
            // pose: [angle, axis_x, axis_y, axis_z, tx, ty, tz]
            Eigen::Matrix<T,3,1> axis(pose[1], pose[2], pose[3]);
            T angle = pose[0];
            Eigen::AngleAxis<T> rotation(angle, axis.normalized());
            Eigen::Translation<T,3> translation(pose[4], pose[5], pose[6]);
            
            Eigen::Matrix<T,3,1> transformed = rotation * current_point.cast<T>() + translation.translation();
            
            // Edge distance calculation
            Eigen::Matrix<T,3,1> line_dir = (map_point2 - map_point1).cast<T>();
            Eigen::Matrix<T,3,1> vec1 = transformed - map_point1.cast<T>();
            Eigen::Matrix<T,3,1> vec2 = transformed - map_point2.cast<T>();
            
            residual[0] = vec1.cross(vec2).norm() / line_dir.norm();
            return true;
        }
        
        Eigen::Vector3f current_point, map_point1, map_point2;
    };

    struct SurfCostFunctor {
        SurfCostFunctor(const Eigen::Vector3f& current,
                       const Eigen::Vector3f& p1,
                       const Eigen::Vector3f& p2,
                       const Eigen::Vector3f& p3)
            : current_point(current), map_point1(p1), map_point2(p2), map_point3(p3) {}
        
        template <typename T>
        bool operator()(const T* const pose, T* residual) const {
            Eigen::Matrix<T,3,1> axis(pose[1], pose[2], pose[3]);
            T angle = pose[0];
            Eigen::AngleAxis<T> rotation(angle, axis.normalized());
            Eigen::Translation<T,3> translation(pose[4], pose[5], pose[6]);
            
            Eigen::Matrix<T,3,1> transformed = rotation * current_point.cast<T>() + translation.translation();
            
            // Plane distance calculation
            Eigen::Matrix<T,3,1> v1 = (map_point2 - map_point1).cast<T>();
            Eigen::Matrix<T,3,1> v2 = (map_point3 - map_point1).cast<T>();
            Eigen::Matrix<T,3,1> normal = v1.cross(v2).normalized();
            
            residual[0] = (transformed - map_point1.cast<T>()).dot(normal);
            return true;
        }
        
        Eigen::Vector3f current_point, map_point1, map_point2, map_point3;
    };

    void estimateOdometry(const FeatureClouds& features) {
        ceres::Problem problem;
        double pose[7] = {0, 0, 0, 1, 0, 0, 0}; // [angle, axis_x, axis_y, axis_z, tx, ty, tz]
        
        // Add edge constraints
        for(const auto& pt : *features.edge) {
            std::vector<int> indices(5);
            std::vector<float> distances(5);
            
            if(map_tree_.nearestKSearch(pt, 5, indices, distances) > 2) {
                Eigen::Vector3f p1 = map_cloud_->points[indices[0]].getVector3fMap();
                Eigen::Vector3f p2 = map_cloud_->points[indices[1]].getVector3fMap();
                
                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<EdgeCostFunctor, 1, 7>(
                        new EdgeCostFunctor(pt.getVector3fMap(), p1, p2));
                problem.AddResidualBlock(cost_function, nullptr, pose);
            }
        }
        
        // Add surface constraints
        for(const auto& pt : *features.surf) {
            std::vector<int> indices(5);
            std::vector<float> distances(5);
            
            if(map_tree_.nearestKSearch(pt, 5, indices, distances) > 3) {
                Eigen::Vector3f p1 = map_cloud_->points[indices[0]].getVector3fMap();
                Eigen::Vector3f p2 = map_cloud_->points[indices[1]].getVector3fMap();
                Eigen::Vector3f p3 = map_cloud_->points[indices[2]].getVector3fMap();
                
                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<SurfCostFunctor, 1, 7>(
                        new SurfCostFunctor(pt.getVector3fMap(), p1, p2, p3));
                problem.AddResidualBlock(cost_function, nullptr, pose);
            }
        }
        
        // Solve optimization problem
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 10;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        // Update pose estimate
        Eigen::AngleAxisf rotation(pose[0], Eigen::Vector3f(pose[1], pose[2], pose[3]).normalized());
        Eigen::Translation3f translation(pose[4], pose[5], pose[6]);
        pose_ = (translation * rotation).matrix() * pose_;
    }

    void updateMap(const FeatureClouds& features) {
        pcl::PointCloud<PointXYZIRT>::Ptr transformed_cloud(new pcl::PointCloud<PointXYZIRT>);
        
        // Transform and add edge features to map
        pcl::transformPointCloud(*features.edge, *transformed_cloud, pose_);
        *map_cloud_ += *transformed_cloud;
        
        // Transform and add surface features to map
        pcl::transformPointCloud(*features.surf, *transformed_cloud, pose_);
        *map_cloud_ += *transformed_cloud;
        
        // Downsample map cloud
        downsampler_.setInputCloud(map_cloud_);
        downsampler_.filter(*map_cloud_);
        
        // Update KD-tree for fast nearest neighbor searches
        map_tree_.setInputCloud(map_cloud_);
        
        // Publish updated map
        sensor_msgs::PointCloud2 map_msg;
        pcl::toROSMsg(*map_cloud_, map_msg);
        map_msg.header.frame_id = "map";
        map_msg.header.stamp = ros::Time::now();
        map_pub_.publish(map_msg);
    }

    void publishOdometry(const ros::Time& stamp) {
        nav_msgs::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = "map";
        odom.child_frame_id = "base_link";
        
        Eigen::Quaternionf q(pose_.block<3,3>(0,0));
        odom.pose.pose.position.x = pose_(0,3);
        odom.pose.pose.position.y = pose_(1,3);
        odom.pose.pose.position.z = pose_(2,3);
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();
        
        odom_pub_.publish(odom);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "loam_odometry_node");
    LOAMOdometry odom;
    ros::spin();
    return 0;
}