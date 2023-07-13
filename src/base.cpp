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

std::mutex mut_img;
std::mutex mut_pc;

// int img_height = 240;
// int img_width = 320;

cv::Mat image_color;
cv::Mat copy_image_color;

pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>); 
pcl::PointCloud<pcl::PointXYZI>::Ptr copy_raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>); 

bool is_rec_image = false;
bool is_rec_LiDAR = false;

typedef pcl::PointXYZRGB PointType; // 색깔점


class ImageLiDARFusion : public rclcpp::Node
{
public:
  pthread_t  tids1_;
  sensor_msgs::msg::PointCloud2 colored_msg;
public:
  cv::Mat transform_matrix;    /**< from globol to image coordinate */
	cv::Mat intrinsic_matrix;    /**< from local to image coordinate  */
	cv::Mat extrinsic_matrix;    /**< from global to local coordinate */
	cv::Mat dist_matrix;         /**< dist parameters  */
  cv::Mat concatedMat;
  
  vector<double> CameraExtrinsic_vector;
  vector<double> CameraMat_vector;
  vector<double> DistCoeff_vector;
  vector<int> ImageSize_vector;

  Mat transformMat;
  Mat CameraExtrinsicMat;
  Mat CameraMat;
  Mat DistCoeff;
  Mat frame1;

  int img_height = 480;
  int img_width = 640;

  float maxlen =200.0;       //maxima distancia del lidar
  float minlen = 0.001;     //minima distancia del lidar
  float max_FOV = CV_PI/4;    // 카메라의 최대 화각(라디안)
public:
  ImageLiDARFusion()
  : Node("Fusion_colored")
  {
    RCLCPP_INFO(this->get_logger(), "----------- initialize -----------\n");

    this->declare_parameter("CameraExtrinsicMat", vector<double>());
    this->CameraExtrinsic_vector = this->get_parameter("CameraExtrinsicMat").as_double_array();
    this->declare_parameter("CameraMat", vector<double>());
    this->CameraMat_vector = this->get_parameter("CameraMat").as_double_array();
    this->declare_parameter("DistCoeff", vector<double>());
    this->DistCoeff_vector = this->get_parameter("DistCoeff").as_double_array();

    this->set_param();

    image_pub = this->create_publisher<sensor_msgs::msg::Image>("corrected_image", 10);
    image_sub = this->create_subscription<sensor_msgs::msg::Image>(
      "video1", 100,
      [this](const sensor_msgs::msg::Image::SharedPtr msg) -> void
      {
        ImageCallback(msg);
      });
    LiDAR_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("test_LiDAR", 10);
    LiDAR_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/velodyne_points", 100,
      [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) -> void
      {
        LiDARCallback(msg);
      });
    
    int ret1 = pthread_create(&this->tids1_, NULL, publish_thread, this);

    RCLCPP_INFO(this->get_logger(), "START LISTENING\n");
  };
  ~ImageLiDARFusion()
  {
  };

  void set_param();
  void LiDARCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  static void * publish_thread(void * this_sub);

private:
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr LiDAR_pub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr LiDAR_sub;
};

void ImageLiDARFusion::set_param()
{
  Mat CameraExtrinsicMat_(4, 4, CV_64F, CameraExtrinsic_vector.data());
  Mat CameraMat_(3, 3, CV_64F, CameraMat_vector.data()); 
  Mat DistCoeffMat_(1, 4, CV_64F, DistCoeff_vector.data());

  CameraExtrinsicMat_.copyTo(this->CameraExtrinsicMat);
  CameraMat_.copyTo(this->CameraMat);
  DistCoeffMat_.copyTo(this->DistCoeff);

  Mat Rlc = CameraExtrinsicMat(cv::Rect(0,0,3,3));
  Mat Tlc = CameraExtrinsicMat(cv::Rect(3,0,1,3));

  cv::hconcat(Rlc, Tlc, this->concatedMat);
  this->transformMat = this->CameraMat * this->concatedMat;
  RCLCPP_INFO(this->get_logger(), "param_ok");
}

void * ImageLiDARFusion::publish_thread(void * args)
{
  ImageLiDARFusion * this_sub = (ImageLiDARFusion *) args;
  rclcpp::WallRate loop_rate(10.0);
  while(rclcpp::ok())
  {
    if (is_rec_image && is_rec_LiDAR)
    {
      mut_pc.lock();
      pcl::copyPointCloud (*raw_pcl_ptr, *copy_raw_pcl_ptr);
      mut_pc.unlock();

      mut_img.lock();
      copy_image_color = image_color.clone();
      mut_img.unlock();

      pcl::PointCloud<PointType>::Ptr pc_xyzrgb(new pcl::PointCloud<PointType>);
      const int size = copy_raw_pcl_ptr->points.size(); 

      cout<<"point cloud size: "<<size<< endl;;

      for(int i =0; i <size; i++)
      {
        // project get the photo coordinate
        // pcl::PointXYZRGB pointRGB;
        PointType pointRGB;

        PointType temp_;
        temp_.x = copy_raw_pcl_ptr->points[i].x;
        temp_.y = copy_raw_pcl_ptr->points[i].y;
        temp_.z = copy_raw_pcl_ptr->points[i].z;
///////////////////////////////////////////////////
        float R_ = sqrt(pow(temp_.x,2)+pow(temp_.y,2));
        // float azimuth_ = abs(atan2(temp_.x, temp_.y)); // output : in radian!!
        float azimuth_ = abs(atan2(temp_.y, temp_.x)); // output : in radian!!
        float elevation_ = abs(atan2(R_, temp_.z)); // 이걸 계산을 이걸로 하는게 맞나? 
/////////////////////////////////////////////////////
        if(azimuth_ > (this_sub->max_FOV) ){
          // cerr << "? ";
          continue;
        }

        pointRGB.x = copy_raw_pcl_ptr->points[i].x;
        pointRGB.y = copy_raw_pcl_ptr->points[i].y;
        pointRGB.z = copy_raw_pcl_ptr->points[i].z;

        double a_[4] = { pointRGB.x, pointRGB.y, pointRGB.z, 1.0 };
        cv::Mat pos(4, 1, CV_64F, a_);

        cv::Mat newpos(this_sub->transformMat * pos);

        float x = (float)(newpos.at<double>(0, 0) / newpos.at<double>(2, 0));
        float y = (float)(newpos.at<double>(1, 0) / newpos.at<double>(2, 0));

        float dist_ = sqrt(pow(pointRGB.x,2) + pow(pointRGB.y,2) + pow(pointRGB.z,2));

        if (this_sub->minlen <  dist_ && dist_ < this_sub->maxlen)
        // if (pointRGB.x != 0)
        {
          if (x >= 0 && x < this_sub->img_width && y >= 0 && y < this_sub->img_height)
          {
            //cout << "3" << endl;
            //  imread BGR（BITMAP）
            int row = int(y);
            int column = int(x);
            // cout << "row: "<<row <<"  / column: "<<column<<endl;
            // cout << "X : " << pointRGB.x << "   / Y : " << pointRGB.y << " /  Z  : " << pointRGB.z << endl;
            // cout << "dist : " << dist_ << endl;
            if (copy_image_color.at<cv::Vec3b>(row, column)[2] == false)
            {
              // cout << "실패!=================================================" << endl;
            }
            pointRGB.r = copy_image_color.at<cv::Vec3b>(row, column)[2];
            pointRGB.g = copy_image_color.at<cv::Vec3b>(row, column)[1];
            pointRGB.b = copy_image_color.at<cv::Vec3b>(row, column)[0];

            pc_xyzrgb->push_back(pointRGB);
          }
        }
      }
      pc_xyzrgb->width = 1;
      pc_xyzrgb->height = pc_xyzrgb->points.size();
      pcl::toROSMsg(*pc_xyzrgb,  this_sub->colored_msg); 
      // this_sub->colored_msg.header.frame_id = "sensor_frame";
      this_sub->colored_msg.header.frame_id = "velodyne";
      this_sub->colored_msg.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
      this_sub->LiDAR_pub->publish(this_sub->colored_msg); 
      loop_rate.sleep();
    }
  }
}

void ImageLiDARFusion::LiDARCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  mut_pc.lock();
  pcl::fromROSMsg(*msg, *raw_pcl_ptr);
  mut_pc.unlock();
  is_rec_LiDAR = true;
}

void ImageLiDARFusion::ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  Mat image;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch(cv_bridge::Exception& e)
  {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  if(IS_IMAGE_CORRECTION)
  {
    mut_img.lock();
    cv::undistort(cv_ptr->image, image_color, this->CameraMat, this->DistCoeff);
    cv_ptr->image = image_color.clone();
    mut_img.unlock();
  }  
  else
  {
    mut_img.lock();
    image_color = cv_ptr->image.clone();
    mut_img.unlock();
  }
  is_rec_image = true;
  sensor_msgs::msg::Image::UniquePtr image_msg = std::make_unique<sensor_msgs::msg::Image>();
  image_msg->height = image_color.rows;
  image_msg->width = image_color.cols;
  image_msg->encoding = "bgr8";
  image_msg->is_bigendian = false;
  image_msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(image_color.step);
  size_t size = image_color.step * image_color.rows;
  image_msg->data.resize(size);
  memcpy(&image_msg->data[0], image_color.data, size);

  image_pub->publish(std::move(image_msg));
  // imshow("image", image_color);
  // waitKey(1);
}



int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImageLiDARFusion>();
  rclcpp::spin(node);
  return 0;
}