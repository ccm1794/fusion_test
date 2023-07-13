#include <rclcpp/rclcpp.hpp>
#include <mutex>

// #include <sensor_msgs/msg/point_cloud.hpp>
// #include <sensor_msgs/msg/point_cloud2.hpp>
//#include <sensor_msgs/point_cloud_conversion.hpp>
#include <sensor_msgs/image_encodings.hpp>

// #include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/msg/point.hpp>
#include <image_transport/image_transport.hpp> //꼭 있을 필요는 없을 듯?
#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//yolo header 추가하기

using namespace std;
using namespace cv;

//typedef pcl::PointXYZRGB PointType; // 색깔점

//static bool IS_IMAGE_CORRECTION = true;

//std::mutex mut_img;
//std::mutex mut_pc;

//포인트클라우드 메시지 : XYZI
//pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>); 
//pcl::PointCloud<pcl::PointXYZI>::Ptr copy_raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>); 

// 카메라 이미지
// cv::Mat image_color;
// cv::Mat copy_image_color;
// cv::Mat img_HSV;

// Yolo Global parameter
// std::mutex mut_yolo;
// std::string Class_name;
int obj_count;
int yolo_num[10][5];

// 이미지, 라이다 들어오는지 확인
// bool is_rec_image = false;
// bool is_rec_LiDAR = false;

class ImageLiDARFusion : public rclcpp::Node
{
public:
  vector<double> CameraExtrinsic_vector;
  vector<double> CameraMat_vector;
  vector<double> DistCoeff_vector;
  vector<int> ImageSize_vector;

  Mat concatedMat;
  Mat transformMat;

  int img_height;
  int img_width;

public:
  ImageLiDARFusion()
  : Node("fusion")
  {
    RCLCPP_INFO(this->get_logger(), "------------ intialize ------------\n");

    this->declare_parameter("CameraExtrinsicMat", vector<double>());
    this->CameraExtrinsic_vector = this->get_parameter("CameraExtrinsicMat").as_double_array();
    this->declare_parameter("CameraMat", vector<double>());
    this->CameraMat_vector = this->get_parameter("CameraMat").as_double_array();
    this->declare_parameter("DistCoeff", vector<double>());
    this->DistCoeff_vector = this->get_parameter("DistCoeff").as_double_array();
    // this->declare_parameter("ImagSize", vector<int>());
    // this->ImageSize_vector = this->get_parameter("ImagSize").as<vector<int>>();

    set_param();

  }
public:
  void set_param();
  //void LiDARCallback(const sensor_msgs::msg::PointCloud2::SharedPtr & msg);
};

void ImageLiDARFusion::set_param()
{
  // 받아온 파라미터
  Mat CameraExtrinsicMat_(4, 4, CV_64F, CameraExtrinsic_vector.data());
  // Mat CameraMat(3, 3, CV_64F, CameraMat_vector.data());
  // Mat DistCoeffMat(1, 5, CV_64F, DistCoeff_vector.data());

  // //재가공 : 회전변환행렬, 평행이동행렬
  // Mat Rlc = CameraExtrinsicMat(cv::Rect(0,0,3,3));
  // Mat Tlc = CameraExtrinsicMat(cv::Rect(3,0,1,3));
  // cv::hconcat(Rlc, Tlc, this->concatedMat);

  // //재가공 : transformMat
  // this->transformMat = CameraMat * this->concatedMat;

  //재가공 : image_size
  // this->img_height = this->ImageSize_vector[0];
  // this->img_width = this->ImageSize_vector[1];
  
  cout << "CameraExtrinsicMat : " << CameraExtrinsicMat_ << endl;
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImageLiDARFusion>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

