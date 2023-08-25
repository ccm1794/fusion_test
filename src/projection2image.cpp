// https://github.com/williamhyin/lidar_to_camera/blob/master/src/project_lidar_to_camera.cpp 투영은 이거 참고함

#include <rclcpp/rclcpp.hpp>
#include <mutex>
#include <memory>
#include <thread>
#include <pthread.h>

#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud_conversion.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/image_encodings.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <image_transport/image_transport.hpp> //꼭 있을 필요는 없을 듯?
#include <cv_bridge/cv_bridge.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//yolo header 추가하기
#include <std_msgs/msg/int16.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

using namespace std;
using namespace cv;

static bool IS_IMAGE_CORRECTION = true;

std::mutex mut_img1, mut_img2;
std::mutex mut_pc;

pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr copy_raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>);

// 카메라 이미지 //
// cv::Mat image_color1, image_color2;
// cv::Mat overlay1, overlay2;
// cv::Mat copy_image_color1, copy_image_color2;

// Yolo Global parameter
std::mutex mut_yolo;
std::string Class_name;

int obj_count;

bool is_rec_LiDAR = false;
int lidar_button;

struct Box_yolo
{
  float x1;
  float x2;
  float y1;
  float y2;
  int color;
};
std::vector<Box_yolo> boxes_yolo;

class ImageLiDARFusion : public rclcpp::Node
{
public:
  vector<double> CameraExtrinsic_vector_right;
  vector<double> CameraExtrinsic_vector_left;

  vector<double> CameraMat_vector_right;
  vector<double> CameraMat_vector_left;

  vector<double> DistCoeff_vector_right;
  vector<double> DistCoeff_vector_left; 

  Mat transformMat_left;
  Mat transformMat_right;

  Mat CameraExtrinsicMat_right;
  Mat CameraExtrinsicMat_left;

  Mat CameraMat_right;
  Mat CameraMat_left;

  Mat DistCoeff_right;
  Mat DistCoeff_left;

  int img_width = 640;
  int img_height = 480;

  float maxlen =200.0;         /**< Max distance: LiDAR */
  float minlen = 0.0001;        /**< Min distance: LiDAR */
  float max_FOV = CV_PI/2;     /**< Max FOV : Camera */

  sensor_msgs::msg::PointCloud2 colored_msg;

  cv::Mat image_color1, image_color2;
  cv::Mat overlay1, overlay2;
  cv::Mat copy_image_color1, copy_image_color2;

public:
  ImageLiDARFusion()
  : Node("projection")
  {
    RCLCPP_INFO(this->get_logger(), "------------ intialize ------------\n");

    this->declare_parameter("CameraExtrinsicMat_right", vector<double>());
    this->CameraExtrinsic_vector_right = this->get_parameter("CameraExtrinsicMat_right").as_double_array();
    this->declare_parameter("CameraMat_right", vector<double>());
    this->CameraMat_vector_right = this->get_parameter("CameraMat_right").as_double_array();

    this->declare_parameter("CameraExtrinsicMat_left", vector<double>());
    this->CameraExtrinsic_vector_left = this->get_parameter("CameraExtrinsicMat_left").as_double_array();
    this->declare_parameter("CameraMat_left", vector<double>());
    this->CameraMat_vector_left = this->get_parameter("CameraMat_left").as_double_array();

    this->set_param();

    image_sub1_ = this->create_subscription<sensor_msgs::msg::Image>(
       "/video1", rclcpp::SensorDataQoS(),
       [this](const sensor_msgs::msg::Image::SharedPtr msg) -> void
       {
         ImageCallback1(msg);
       }); //람다함수를 사용했는데 왜?, 그리고 일반적인 방법으로 하면 동작하지 않는다. 이유를 모르겠다
    
    image_sub2_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/video2", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::Image::SharedPtr msg) -> void
      {
        ImageCallback2(msg);
      });

    LiDAR_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/velodyne_points", 10,
      [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) -> void
      {
        LiDARCallback(msg);
      });

    auto timer_callback = [this]() -> void {ProjectionCallback();};
    timer_ = create_wall_timer(100ms, timer_callback);

    RCLCPP_INFO(this->get_logger(), "------------ intialize end------------\n");
  }

  ~ImageLiDARFusion(){}

public:
  void set_param();
  void ImageCallback1(const sensor_msgs::msg::Image::SharedPtr msg);
  void ImageCallback2(const sensor_msgs::msg::Image::SharedPtr msg);
  void LiDARCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void ProjectionCallback();

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub1_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub2_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr LiDAR_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};


void ImageLiDARFusion::set_param()
{
  // right //
  Mat CameraExtrinsicMat_right_(3, 4, CV_64F, CameraExtrinsic_vector_right.data());
  Mat CameraMat_right_(3, 3, CV_64F, CameraMat_vector_right.data()); 

  CameraExtrinsicMat_right_.copyTo(this->CameraExtrinsicMat_right);
  CameraMat_right_.copyTo(this->CameraMat_right);
  
  //재가공 : transformMat_right
  this->transformMat_right = this->CameraMat_right * this->CameraExtrinsicMat_right;

  // left //
  Mat CameraExtrinsicMat_left_(3, 4, CV_64F, CameraExtrinsic_vector_left.data());
  Mat CameraMat_left_(3, 3, CV_64F, CameraMat_vector_left.data());

  CameraExtrinsicMat_left_.copyTo(this->CameraExtrinsicMat_left);
  CameraMat_left_.copyTo(this->CameraMat_left);

  //재가공 : transformMat_left
  this->transformMat_left = this->CameraMat_left * this->CameraExtrinsicMat_left;

  RCLCPP_INFO(this->get_logger(), "ok");
}

void ImageLiDARFusion::ImageCallback1(const sensor_msgs::msg::Image::SharedPtr msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  Mat image;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
  }
  catch(cv_bridge::Exception& e)
  {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  mut_img1.lock();
  cv::undistort(cv_ptr->image, image_color1, this->CameraMat_right, this->DistCoeff_right);
  image_color1 = image_color1.clone();
  overlay1 = image_color1.clone();
  mut_img1.unlock();
}

void ImageLiDARFusion::ImageCallback2(const sensor_msgs::msg::Image::SharedPtr msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  Mat image;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
  }
  catch(cv_bridge::Exception& e)
  {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  mut_img2.lock();
  cv::undistort(cv_ptr->image, image_color2, this->CameraMat_left, this->DistCoeff_left);
  image_color2 = image_color2.clone();
  overlay2 = image_color2.clone();
  mut_img2.unlock();
}

void ImageLiDARFusion::LiDARCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  mut_pc.lock();
  pcl::fromROSMsg(*msg, *raw_pcl_ptr);
  mut_pc.unlock();
}

void ImageLiDARFusion::ProjectionCallback()
{
  mut_img1.lock();
  copy_image_color1 = image_color1.clone();
  mut_img1.unlock();

  mut_img2.lock();
  copy_image_color2 = image_color2.clone();
  mut_img2.unlock();

  mut_pc.lock();
  pcl::copyPointCloud(*raw_pcl_ptr, *copy_raw_pcl_ptr);
  mut_pc.unlock();

  // Mat concatedMat = cv::hconcat(copy_image_color1, copy_image_color2);

  const int size = copy_raw_pcl_ptr->points.size();
  
  for(int i = 0; i < size ; i++)
  {
    pcl::PointXYZ temp_;
    temp_.x = copy_raw_pcl_ptr->points[i].x;
    temp_.y = copy_raw_pcl_ptr->points[i].y;
    temp_.z = copy_raw_pcl_ptr->points[i].z;

    float azimuth_ = atan2(temp_.y, temp_.x); // x축 기준으로 방위각 부호가 바뀜

    if (abs(azimuth_) > max_FOV)
    {
      continue;
    }
    else if(azimuth_ > -10 * (CV_PI/180) && azimuth_ < (CV_PI/2)) // y축이 양수일 때(erp기준 오른쪽 카메라)
    {
      double a_[4] = {temp_.x, temp_.y, temp_.z, 1.0};
      cv::Mat pos( 4, 1, CV_64F, a_); // 라이다 좌표
      cv::Mat newpos(transformMat_right * pos); // 카메라 좌표로 변환한 것.

      float x = (float)(newpos.at<double>(0, 0) / newpos.at<double>(2, 0));
      float y = (float)(newpos.at<double>(1, 0) / newpos.at<double>(2, 0));

      float dist_ = sqrt(pow(temp_.x,2) + pow(temp_.y,2) + pow(temp_.z,2));

      if (this->minlen < dist_ && this->maxlen > dist_)
      {
        if(x >= 0 && x < this->img_width && y >= 0 && y < this->img_height)
        {
          int row = int(y);
          int column = int(x);

          cv::Point pt;
          pt.x = x;
          pt.y = y;

          float val = temp_.x;
          float maxVal = 200.0;

          int green = min(255, (int) (255 * abs((val - maxVal) / maxVal)));
          int red = min(255, (int) (255 * (1 - abs((val - maxVal) / maxVal))));
          cv::circle(overlay1, pt, 2, cv::Scalar(0, green, red), -1);
        }
      }
    }
    else if(azimuth_ < 10*(CV_PI/180) && azimuth_ > -(CV_PI/2))
    {
      double a_[4] = {temp_.x, temp_.y, temp_.z, 1.0};
      cv::Mat pos( 4, 1, CV_64F, a_); // 라이다 좌표
      cv::Mat newpos(transformMat_left * pos); // 카메라 좌표로 변환한 것.

      float x = (float)(newpos.at<double>(0, 0) / newpos.at<double>(2, 0));
      float y = (float)(newpos.at<double>(1, 0) / newpos.at<double>(2, 0));

      float dist_ = sqrt(pow(temp_.x,2) + pow(temp_.y,2) + pow(temp_.z,2));   

      if (this->minlen < dist_ && this->maxlen > dist_)
      {
        if(x >= 0 && x < this->img_width && y >= 0 && y < this->img_height)
        {
          int row = int(y);
          int column = int(x);

          cv::Point pt;
          pt.x = x;
          pt.y = y;

          float val = temp_.x;
          float maxVal = 200.0;

          int green = min(255, (int) (255 * abs((val - maxVal) / maxVal)));
          int red = min(255, (int) (255 * (1 - abs((val - maxVal) / maxVal))));
          cv::circle(overlay2, pt, 2, cv::Scalar(0, green, red), -1);
        }
      }   
    }
  
    float opacity = 0.6;
    cv::addWeighted(overlay1, opacity, copy_image_color1, 1 - opacity, 0, copy_image_color1);
    cv::addWeighted(overlay2, opacity, copy_image_color2, 1 - opacity, 0, copy_image_color2);

    string windowName_right = "overlay_right";
    string windowName_left = "overlay_left";
    cv::namedWindow(windowName_right, 3);
    cv::namedWindow(windowName_left, 3);
    cv::imshow(windowName_right, copy_image_color1);
    cv::imshow(windowName_left, copy_image_color2);
    char ch = cv::waitKey(10);

  }
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImageLiDARFusion>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}