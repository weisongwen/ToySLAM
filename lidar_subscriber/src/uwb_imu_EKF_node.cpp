#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PointStamped.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>

class ESKFFusionNode {
public:
    ESKFFusionNode() : nh_("~") {
        // Subscribers
        // imu_sub_ = nh_.subscribe("/sensor_simulator/imu_data", 100, &ESKFFusionNode::imuCallback, this);
        // uwb_sub_ = nh_.subscribe("/sensor_simulator/UWBPoistionPS", 10, &ESKFFusionNode::uwbCallback, this);

        imu_sub_ = nh_.subscribe("/imu/data", 100, &ESKFFusionNode::imuCallback, this);
        uwb_sub_ = nh_.subscribe("/vins_estimator/UWBPoistionPS", 10, &ESKFFusionNode::uwbCallback, this);

        user_path_pub_ = nh_.advertise<nav_msgs::Path>("user_path", 10);
        
        // Publisher
        fused_pose_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/fusedPos", 10);
        
        // Initialize state and covariance
        initState();
        
        // Noise parameters (tune these)
        acc_noise_ = 0.01;
        gyro_noise_ = 0.005;
        acc_bias_noise_ = 0.0001;
        gyro_bias_noise_ = 0.0001;
        
        uwb_noise_ = 0.001; // 0.1

        last_imu_time_ = ros::Time::now();
    }

    /* initialize the state vector */
    void initState() {
        // State vector: [position(3), velocity(3), orientation(4), acc_bias(3), gyro_bias(3)]
        x_ = Eigen::VectorXd::Zero(16);
        x_.segment<4>(6) = Eigen::Vector4d(1, 0, 0, 0); // Identity quaternion
        
        // Covariance matrix (15x15 error state)
        // P_ = Eigen::MatrixXd::Identity(15, 15) * 0.1;
        P_ = Eigen::MatrixXd::Identity(15, 15) * 0.1;
    }

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        // Time handling
        ros::Time current_time = msg->header.stamp;
        double dt = (current_time - last_imu_time_).toSec();
        last_imu_time_ = current_time;
        if(dt <= 0) return;

        // IMU measurements
        Eigen::Vector3d acc(msg->linear_acceleration.x,
                            msg->linear_acceleration.y,
                            msg->linear_acceleration.z);
        Eigen::Vector3d gyro(msg->angular_velocity.x,
                             msg->angular_velocity.y,
                             msg->angular_velocity.z);
        
        // std::cout<<"acc->  "<< acc<<"\n";
        // std::cout<<"gyro->  "<< gyro<<"\n";
        // Prediction step
        predict(acc, gyro, dt);
    }

    void uwbCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
        // UWB measurement
        Eigen::Vector3d z(msg->point.x, msg->point.y, msg->point.z);

        // std::cout<<"z->  "<< z<<"\n";
        
        // Update step
        update(z);
        
        // Publish fused pose
        publishFusedPose(msg->header.stamp);

        Eigen::Vector3d p = x_.segment<3>(0);
        std::cout<<"p->  "<< p<<"\n";
    }

private:
    void predict(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro, double dt) {
        // Extract states
        Eigen::Vector3d p = x_.segment<3>(0);
        Eigen::Vector3d v = x_.segment<3>(3);
        Eigen::Quaterniond q(x_.segment<4>(6).data());
        Eigen::Vector3d b_a = x_.segment<3>(10);
        Eigen::Vector3d b_g = x_.segment<3>(13);

        // Bias corrected measurements
        Eigen::Vector3d acc_unbias = acc - b_a;
        Eigen::Vector3d gyro_unbias = gyro - b_g;

        // Orientation update
        Eigen::Quaterniond dq = Eigen::Quaterniond::Identity();
        Eigen::Vector3d omega = gyro_unbias * dt;
        double theta = omega.norm();
        if(theta > 1e-6) {
            Eigen::Vector3d axis = omega / theta;
            dq = Eigen::Quaterniond(Eigen::AngleAxisd(theta, axis));
        }
        q = (q * dq).normalized();

    
        // Acceleration in world frame
        Eigen::Vector3d a_world = q * acc_unbias - Eigen::Vector3d(0, 0, 9.81);

        // Position and velocity update
        v += a_world * dt;
        p += v * dt + 0.5 * a_world * dt * dt;

        // Update state
        x_.segment<3>(0) = p;
        x_.segment<3>(3) = v;
        x_.segment<4>(6) = Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
        
        // Update covariance (simplified)
        Eigen::MatrixXd F = computeF(q, a_world, dt);
        if (omega.norm() > 1e-12) {
        F.block<3, 3>(6, 6) = Eigen::AngleAxisd(omega.norm(), omega.normalized()).toRotationMatrix().transpose();
        } else {
            F.block<3, 3>(6, 6).setIdentity();
        }
        
        Eigen::MatrixXd Q = computeQ(dt);
        // Eigen::MatrixXd Q = computeQV2(dt);
        // std::cout<<"theta->  "<< theta<<"\n";
        P_ = F * P_ * F.transpose() + Q;

        // std::cout<<"predicted p->  "<< p<<"\n";
    }

    Eigen::MatrixXd computeF(const Eigen::Quaterniond& q, const Eigen::Vector3d& a_world, double dt) {
        // Simplified state transition Jacobian (15x15)
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(15, 15);
        
        // Position derivatives
        F.block<3,3>(0,3) = Eigen::Matrix3d::Identity() * dt;
        
        // Velocity derivatives
        F.block<3,3>(3,6) = -q.toRotationMatrix() * skew(a_world) * dt;
        // F.block<3,3>(3,10) = -q.toRotationMatrix() * dt; // this is a bug, refer to https://github.dev/ydsf16/imu_gps_localization/tree/master/imu_gps_localizer/include/imu_gps_localizer
        F.block<3,3>(3,9) = -q.toRotationMatrix() * dt;
        
        // Orientation derivatives
        F.block<3,3>(6,6) = Eigen::Matrix3d::Identity(); // Simplified ???

        F.block<3, 3>(6, 12)  = - Eigen::Matrix3d::Identity() * dt;
        
        return F;
    }

    Eigen::MatrixXd computeQ(double dt) {
        // Process noise covariance
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(15, 15);
        // Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(16, 16);
        Q.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * pow(acc_noise_, 2) * pow(dt, 4);
        Q.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * pow(acc_noise_, 2) * pow(dt, 2);
        Q.block<3,3>(6,6) = Eigen::Matrix3d::Identity() * pow(gyro_noise_, 2) * pow(dt, 2);

        // Q.block<3,3>(10,10) = Eigen::Matrix3d::Identity() * acc_bias_noise_ * dt;
        // Q.block<3,3>(13,13) = Eigen::Matrix3d::Identity() * gyro_bias_noise_ * dt;

        Q.block<3,3>(9,9) = Eigen::Matrix3d::Identity() * acc_bias_noise_ * dt;
        Q.block<3,3>(12,12) = Eigen::Matrix3d::Identity() * gyro_bias_noise_ * dt;
        return Q;
    }

    Eigen::MatrixXd computeQV2(double dt) {
        double dt2 = dt * dt;
        Eigen::Matrix<double, 15, 12> Fi = Eigen::Matrix<double, 15, 12>::Zero();
        Fi.block<12, 12>(3, 0) = Eigen::Matrix<double, 12, 12>::Identity();

        Eigen::Matrix<double, 12, 12> Qi = Eigen::Matrix<double, 12, 12>::Zero();
        Qi.block<3, 3>(0, 0) = dt2 * acc_noise_ * Eigen::Matrix3d::Identity();
        Qi.block<3, 3>(3, 3) = dt2 * gyro_noise_ * Eigen::Matrix3d::Identity();
        Qi.block<3, 3>(6, 6) = dt * acc_bias_noise_ * Eigen::Matrix3d::Identity();
        Qi.block<3, 3>(9, 9) = dt * gyro_bias_noise_ * Eigen::Matrix3d::Identity();
        return Fi * Qi * Fi.transpose();
    }

    void update(const Eigen::Vector3d& z) {
        // Measurement matrix (3x15)
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 15);
        H.block<3,3>(0,0) = Eigen::Matrix3d::Identity();

        // Measurement noise
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * uwb_noise_;

        // Kalman gain
        Eigen::MatrixXd S = H * P_ * H.transpose() + R;
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();

        // Update error state
        Eigen::Vector3d error = z - x_.segment<3>(0);
        Eigen::VectorXd dx = K * error;

        // Update nominal state
        x_.segment<3>(0) += dx.segment<3>(0);
        x_.segment<3>(3) += dx.segment<3>(3);
        updateOrientation(dx.segment<3>(6));
        x_.segment<3>(10) += dx.segment<3>(9);
        x_.segment<3>(13) += dx.segment<3>(12);
        // x_.segment<3>(9) += dx.segment<3>(9);
        // x_.segment<3>(12) += dx.segment<3>(12);

        // Update covariance
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(15, 15);
        P_ = (I - K * H) * P_;
    }

    void updateOrientation(const Eigen::Vector3d& delta_theta) {
        Eigen::Quaterniond q(x_.segment<4>(6).data());
        Eigen::Quaterniond dq = Eigen::Quaterniond(1, 
                                                0.5 * delta_theta.x(),
                                                0.5 * delta_theta.y(),
                                                0.5 * delta_theta.z()).normalized();
        q = (q * dq).normalized();
        x_.segment<4>(6) = Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
    }

    Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d S;
        S << 0, -v.z(), v.y(),
             v.z(), 0, -v.x(),
             -v.y(), v.x(), 0;
        return S;
    }

    void publishFusedPose(const ros::Time& stamp) {
        geometry_msgs::PointStamped pose_msg;
        pose_msg.header.stamp = stamp;
        pose_msg.header.frame_id = "map";
        pose_msg.point.x = x_(0);
        pose_msg.point.y = x_(1);
        pose_msg.point.z = x_(2);
        fused_pose_pub_.publish(pose_msg);

        // Add to path
        geometry_msgs::PoseStamped pose;
        pose.header.stamp = stamp;
        pose.header.frame_id = "map";

        path_user_position.header.stamp = stamp;
        path_user_position.header.frame_id = "map";

        pose.pose.position.x = x_(0);
        pose.pose.position.y = x_(1);
        pose.pose.position.z = x_(2);
        path_user_position.poses.push_back(pose);
        user_path_pub_.publish(path_user_position);

    }

    ros::NodeHandle nh_;
    ros::Subscriber imu_sub_, uwb_sub_;
    ros::Publisher fused_pose_pub_;
    ros::Publisher user_path_pub_;
    nav_msgs::Path path_user_position;
    
    Eigen::VectorXd x_;      // 16D state vector
    Eigen::MatrixXd P_;      // 15x15 covariance matrix
    
    // Noise parameters
    double acc_noise_;
    double gyro_noise_;
    double acc_bias_noise_;
    double gyro_bias_noise_;
    double uwb_noise_;
    
    ros::Time last_imu_time_;
};

/*
main function
 */
int main(int argc, char** argv) {
    ros::init(argc, argv, "eskf_fusion_node");
    ESKFFusionNode node;
    ros::spin();
    return 0;
}