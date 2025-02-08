#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Dense>
#include <vector>

class GPSRayTracer {
private:
    ros::NodeHandle nh;
    ros::Publisher marker_pub;
    
    struct Building {
        Eigen::Vector3d position;
        Eigen::Vector3d size;
        Eigen::Matrix3d rotation;
    };
    
    std::vector<Building> buildings;
    std::vector<Eigen::Vector3d> transmitters;

public:
    GPSRayTracer() {
        marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);
        initializeSimulation();
    }

    void initializeSimulation() {
        // Create sample buildings
        buildings.push_back({
            Eigen::Vector3d(0, 0, 15), 
            Eigen::Vector3d(20, 30, 30),
            Eigen::Matrix3d::Identity()
        });

        // Create sample GPS transmitters (satellites)
        transmitters.push_back(Eigen::Vector3d(-100, 50, 200));
        transmitters.push_back(Eigen::Vector3d(150, -80, 180));
        transmitters.push_back(Eigen::Vector3d(80, 120, 190));
    }

    bool rayBuildingIntersection(const Eigen::Vector3d& origin, const Eigen::Vector3d& dir,
                                 const Building& building, Eigen::Vector3d& intersection) {
        // Simplified AABB intersection test
        Eigen::Vector3d min = building.position - building.size/2;
        Eigen::Vector3d max = building.position + building.size/2;
        
        double tmin = (min.x() - origin.x()) / dir.x(); 
        double tmax = (max.x() - origin.x()) / dir.x(); 
        if (tmin > tmax) std::swap(tmin, tmax); 

        double tymin = (min.y() - origin.y()) / dir.y(); 
        double tymax = (max.y() - origin.y()) / dir.y(); 
        if (tymin > tymax) std::swap(tymin, tymax); 

        if ((tmin > tymax) || (tymin > tmax)) 
            return false; 

        if (tymin > tmin) tmin = tymin; 
        if (tymax < tmax) tmax = tymax; 

        double tzmin = (min.z() - origin.z()) / dir.z(); 
        double tzmax = (max.z() - origin.z()) / dir.z(); 
        if (tzmin > tzmax) std::swap(tzmin, tzmax); 

        if ((tmin > tzmax) || (tzmin > tmax)) 
            return false; 

        if (tzmin > tmin) tmin = tzmin; 
        if (tzmax < tmax) tmax = tzmax; 

        if (tmax < 0) return false;

        intersection = origin + dir * tmin;
        return true;
    }

    void visualize() {
        visualization_msgs::Marker building_marker, ray_marker;
        
        // Building visualization
        building_marker.header.frame_id = "map";
        building_marker.header.stamp = ros::Time::now();
        building_marker.ns = "buildings";
        building_marker.action = visualization_msgs::Marker::ADD;
        building_marker.type = visualization_msgs::Marker::CUBE_LIST;
        building_marker.scale.x = 1.0;
        building_marker.scale.y = 1.0;
        building_marker.scale.z = 1.0;
        building_marker.color.a = 0.7;
        building_marker.color.r = 0.5;
        building_marker.color.g = 0.5;
        building_marker.color.b = 0.5;

        // Ray visualization
        ray_marker = building_marker;
        ray_marker.ns = "rays";
        ray_marker.type = visualization_msgs::Marker::LINE_LIST;
        ray_marker.scale.x = 0.1;
        ray_marker.color.a = 1.0;
        ray_marker.color.r = 1.0;
        ray_marker.points.clear();

        // Add buildings to marker
        for (const auto& b : buildings) {
            geometry_msgs::Point p;
            p.x = b.position.x();
            p.y = b.position.y();
            p.z = b.position.z();
            building_marker.points.push_back(p);
            building_marker.scale.x = b.size.x();
            building_marker.scale.y = b.size.y();
            building_marker.scale.z = b.size.z();
        }

        // Process rays
        Eigen::Vector3d receiver(0, 0, 0); // Ground receiver
        for (const auto& tx : transmitters) {
            Eigen::Vector3d direction = (receiver - tx).normalized();
            Eigen::Vector3d intersection;
            
            // Direct path
            bool blocked = false;
            for (const auto& b : buildings) {
                if (rayBuildingIntersection(tx, direction, b, intersection)) {
                    blocked = true;
                    break;
                }
            }
            
            if (!blocked) {
                geometry_msgs::Point p1, p2;
                p1.x = tx.x(); p1.y = tx.y(); p1.z = tx.z();
                p2.x = receiver.x(); p2.y = receiver.y(); p2.z = receiver.z();
                ray_marker.points.push_back(p1);
                ray_marker.points.push_back(p2);
            }

            // Reflected paths
            for (const auto& b : buildings) {
                if (rayBuildingIntersection(tx, direction, b, intersection)) {
                    Eigen::Vector3d normal = (intersection - b.position).normalized();
                    Eigen::Vector3d reflect_dir = direction - 2 * direction.dot(normal) * normal;
                    
                    geometry_msgs::Point p1, p2, p3;
                    p1.x = tx.x(); p1.y = tx.y(); p1.z = tx.z();
                    p2.x = intersection.x(); p2.y = intersection.y(); p2.z = intersection.z();
                    p3.x = receiver.x(); p3.y = receiver.y(); p3.z = receiver.z();
                    
                    ray_marker.points.push_back(p1);
                    ray_marker.points.push_back(p2);
                    ray_marker.points.push_back(p2);
                    ray_marker.points.push_back(p3);
                }
            }
        }

        marker_pub.publish(building_marker);
        marker_pub.publish(ray_marker);
    }

    void run() {
        ros::Rate rate(1);
        while (ros::ok()) {
            visualize();
            rate.sleep();
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "gps_ray_tracer");
    GPSRayTracer tracer;
    tracer.run();
    return 0;
}