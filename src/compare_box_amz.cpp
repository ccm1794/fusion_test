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

// yolo header 추가하기
#include <std_msgs/msg/int16.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/bounding_box3_d.hpp>

using namespace std;
using namespace cv;

// 클러스터 박스 좌표를 저장하기 위한 구조체
struct Box
{
  float x;
  float y;
  float z;

  float size_x;
  float size_y;
  float size_z;
};

// 클러스터 박스 처리를 위한 구조체
struct Box_points
{
  float x1, y1, z1;
  float x2, y2, z2;
  float x3, y3, z3;
  float x4, y4, z4;
};

// 욜로 박스 저장할 구조체
struct Box_yolo
{
  float x1;
  float x2;
  float y1;
  float y2;
  int color;
};

// 클러스터 박스를 크기순대로 정렬하기 위한 함수
bool compareBoxes(const Box &a, const Box &b)
{
  return a.x < b.x;
}

// 클러스터 박스의 8개의 꼭지점을 구하는 함수
Box_points calcBox_points(const Box &box)
{
  Box_points vertices;
  if (box.y >= 0)
  {
    vertices.x1 = box.x - (box.size_x) / 2;
    vertices.y1 = box.y + (box.size_y) / 2;
    vertices.z1 = box.z - (box.size_z) / 2;

    vertices.x2 = box.x + (box.size_x) / 2;
    vertices.y2 = box.y - (box.size_y) / 2;
    vertices.z2 = box.z - (box.size_z) / 2;

    vertices.x3 = box.x + (box.size_x) / 2;
    vertices.y3 = box.y - (box.size_y) / 2;
    vertices.z3 = box.z + (box.size_z) / 2;

    vertices.x4 = box.x - (box.size_x) / 2;
    vertices.y4 = box.y + (box.size_y) / 2;
    vertices.z4 = box.z + (box.size_z) / 2;
  }
  else if (box.y < 0)
  {
    vertices.x1 = box.x + (box.size_x) / 2;
    vertices.y1 = box.y + (box.size_y) / 2;
    vertices.z1 = box.z - (box.size_z) / 2;

    vertices.x2 = box.x - (box.size_x) / 2;
    vertices.y2 = box.y - (box.size_y) / 2;
    vertices.z2 = box.z - (box.size_z) / 2;

    vertices.x3 = box.x - (box.size_x) / 2;
    vertices.y3 = box.y - (box.size_y) / 2;
    vertices.z3 = box.z + (box.size_z) / 2;

    vertices.x4 = box.x + (box.size_x) / 2;
    vertices.y4 = box.y + (box.size_y) / 2;
    vertices.z4 = box.z + (box.size_z) / 2;
  }
  return vertices;
}

// iou 계산
float get_iou(const Box_yolo &a, const Box_yolo &b)
{
  float x_left = max(a.x1, b.x1);
  float y_top = max(a.y1, b.y1);
  float x_right = min(a.x2, b.x2);
  float y_bottom = min(a.y2, b.y2);

  if (x_right < x_left || y_bottom < y_top)
    return 0.0f;

  float intersection_area = (x_right - x_left) * (y_bottom - y_top);

  float area1 = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area2 = (b.x2 - b.x1) * (b.y2 - b.y1);

  float iou = intersection_area / (area1 + area2 - intersection_area);

  return iou;
}

std::mutex mut_img, mut_pc, mut_box;
std::vector<Box> boxes;
std::vector<Box_yolo> boxes_yolo;

cv::Mat image_color;
cv::Mat copy_image_color;

std::mutex mut_yolo;
std::string Class_name;

class ImageLiDARFusion : public rclcpp::Node
{
public:
  vector<double> CameraExtrinsic_vector_right;
  vector<double> CameraExtrinsic_vector_left;
  vector<double> CameraMat_vector_right;
  vector<double> CameraMat_vector_left;
  vector<double> DistCoeff_vector_right;
  vector<double> DistCoeff_vector_left;

  Mat transformMat_right;
  Mat transformMat_left;

  bool is_rec_image = false;
  bool is_rec_yolo = false;
  bool is_rec_box = false;

  int box_locker = 0; // box_Callback 할 때 lock해주는 변수
  int yolo_locker = 0;
  // int cluster_count = 0;

  std_msgs::msg::Float64MultiArray cone_msg; // to planning
  sensor_msgs::msg::PointCloud2 pointcloud_msg; // to rviz

public:
  ImageLiDARFusion()
  : Node("amz_compare_box")
  {
    RCLCPP_INFO(this->get_logger(), "----------- initialize -----------\n");

    this->declare_parameter("CameraExtrinsicMat_right", vector<double>());
    this->CameraExtrinsic_vector_right = this->get_parameter("CameraExtrinsicMat_right").as_double_array();
    this->declare_parameter("CameraMat_right", vector<double>());
    this->CameraMat_vector_right = this->get_parameter("CameraMat_right").as_double_array();
    this->declare_parameter("DistCoeff_right", vector<double>());
    this->DistCoeff_vector_right = this->get_parameter("DistCoeff_right").as_double_array();   

    this->declare_parameter("CameraExtrinsicMat_left", vector<double>());
    this->CameraExtrinsic_vector_left = this->get_parameter("CameraExtrinsicMat_left").as_double_array();
    this->declare_parameter("CameraMat_left", vector<double>());
    this->CameraMat_vector_left = this->get_parameter("CameraMat_left").as_double_array();
    this->declare_parameter("DistCoeff_left", vector<double>());
    this->DistCoeff_vector_left = this->get_parameter("DistCoeff_left").as_double_array();   

    this->set_param();

    LiDAR_sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
      "/lidar_boxes", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
      [this](const vision_msgs::msg::Detection3DArray::SharedPtr msg) -> void
      {
        BoxCallback(msg);
      });

    yolo_detect_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
      "/yolo_detect", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
      [this](const vision_msgs::msg::Detection2DArray::SharedPtr msg) -> void
      {
        YOLOCallback(msg);
      });

    LiDAR_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("test_LiDAR", rclcpp::SensorDataQoS());
    cone_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/LiDAR/center_color", rclcpp::SensorDataQoS());

    // 타이머 콜백으로 박스 매칭함수 실행
    auto timer_callback = [this]() -> void{ FusionCallback(); };
    timer_ = create_wall_timer(50ms, timer_callback); // 10hz

    RCLCPP_INFO(this->get_logger(), "------------ intialize end------------\n");
  }

  ~ImageLiDARFusion(){}

  void set_param();
  void BoxCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);
  void YOLOCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);
  void FusionCallback();

private:
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr LiDAR_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr yolo_detect_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr LiDAR_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cone_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

void ImageLiDARFusion::set_param()
{
  // 받아온 파라미터
  Mat CameraExtrinsicMat_right(3, 4, CV_64F, CameraExtrinsic_vector_right.data());
  Mat CameraExtrinsicMat_left(3, 4, CV_64F, CameraExtrinsic_vector_left.data());
  Mat CameraMat_right(3, 3, CV_64F, CameraMat_vector_right.data());
  Mat CameraMat_left(3, 3, CV_64F, CameraMat_vector_left.data());
  Mat DistCoeffMat_right(1,4, CV_64F, DistCoeff_vector_right.data());
  Mat DistCoeffMat_left(1,4, CV_64F, DistCoeff_vector_left.data());

  // 재가공 : transformMat
  this->transformMat_right = CameraMat_right * CameraExtrinsicMat_right;
  this->transformMat_left = CameraMat_left * CameraExtrinsicMat_left;

  cout << "transformMat_right : " << this->transformMat_right << "\n" << "transformMat_left : " << this->transformMat_left << "\n";
  RCLCPP_INFO(this->get_logger(), "parameter set end");
}

void ImageLiDARFusion::BoxCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
{
  // this->cluster_count = msg->detections.size();
  if (msg->detections.size() != 0 && this->box_locker == 0)
  {
    // mut_box.lock();
    this->box_locker = 1;
    for (int i = 0; i < msg->detections.size(); i++)
    {
      if (msg->detections[i].bbox.size.y < 0.5 && msg->detections[i].bbox.size.z < 1)
      {
        mut_box.lock();
        Box box =
        {
            msg->detections[i].bbox.center.position.x,
            msg->detections[i].bbox.center.position.y,
            msg->detections[i].bbox.center.position.z,
            msg->detections[i].bbox.size.x,
            msg->detections[i].bbox.size.y * 1.2, // 투영되는 박스 크기를 임의로 변형
            msg->detections[i].bbox.size.z * 1.2  // 투영되는 박스 크기를 임의로 변형
        };
        boxes.push_back(box);
        mut_box.unlock();
      }
    }
    // mut_box.unlock();
    this->is_rec_box = true;
  }
}

void ImageLiDARFusion::YOLOCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
  // int obj_count = msg->detections.size();

  if (msg->detections.size() != 0 && this->yolo_locker == 0)
  {
    this->yolo_locker = 1;
    std::string Class_name;

    // mut_yolo.lock();
    for (int i = 0; i < msg->detections.size(); i++)
    {
      Class_name = msg->detections[i].results[0].id;

      int color = 0;
      if (Class_name == "blue")
      {
          color = 1;
      }
      else if (Class_name == "orange")
      {
          color = 2;
      }
      else
      {
          color = 0;
      }
      mut_yolo.lock();
      Box_yolo box_yolo =
      {
        msg->detections[i].bbox.center.x - (msg->detections[i].bbox.size_x) / 2, // x1
        msg->detections[i].bbox.center.x + (msg->detections[i].bbox.size_x) / 2, // x2
        msg->detections[i].bbox.center.y - (msg->detections[i].bbox.size_y) / 2, // y1
        msg->detections[i].bbox.center.y + (msg->detections[i].bbox.size_y) / 2, // y2
        color
      };
      boxes_yolo.push_back(box_yolo);
      mut_yolo.unlock();
    }
    this->is_rec_yolo = true;
    // mut_yolo.unlock();
  }
  else
  {
  }
}

void ImageLiDARFusion::FusionCallback()
{
  mut_box.lock();
  std::vector<Box> lidar_boxes = boxes;
  mut_box.unlock();

  mut_yolo.lock();
  std::vector<Box_yolo> yolo_boxes = boxes_yolo;
  mut_yolo.unlock();

  // pcl::PointCloud<pcl::PointXYZI>::Ptr coord_cloud(new pcl::PointCloud<pcl::PointXYZI>);

  if (this->yolo_locker == 1 && this->box_locker == 1)
  {
    std::vector<Box_yolo> boxes_2d_cluster;

    for (const auto &Box : lidar_boxes)
    {
      Box_points vertices = calcBox_points(Box);
      double box_1[4] = {vertices.x1, vertices.y1, vertices.z1, 1.0};
      double box_2[4] = {vertices.x2, vertices.y2, vertices.z2, 1.0};
      double box_3[4] = {vertices.x3, vertices.y3, vertices.z3, 1.0};
      double box_4[4] = {vertices.x4, vertices.y4, vertices.z4, 1.0};

      cv::Mat pos1(4, 1, CV_64F, box_1); // 3차원 좌표
      cv::Mat pos2(4, 1, CV_64F, box_2);
      cv::Mat pos3(4, 1, CV_64F, box_3);
      cv::Mat pos4(4, 1, CV_64F, box_4);

      if (Box.y >= 0)
      {
        // 카메라 원점 xyz 좌표 (3,1)생성
        cv::Mat newpos1(this->transformMat_left * pos1); // 카메라 좌표로 변환한 것.
        cv::Mat newpos2(this->transformMat_left * pos2);
        cv::Mat newpos3(this->transformMat_left * pos3);
        cv::Mat newpos4(this->transformMat_left * pos4);

        float x1 = (float)(newpos1.at<double>(0, 0) / newpos1.at<double>(2, 0));
        float y1 = (float)(newpos1.at<double>(1, 0) / newpos1.at<double>(2, 0));

        float x2 = (float)(newpos2.at<double>(0, 0) / newpos2.at<double>(2, 0));
        float y2 = (float)(newpos2.at<double>(1, 0) / newpos2.at<double>(2, 0));

        float x3 = (float)(newpos3.at<double>(0, 0) / newpos3.at<double>(2, 0));
        float y3 = (float)(newpos3.at<double>(1, 0) / newpos3.at<double>(2, 0));

        float x4 = (float)(newpos4.at<double>(0, 0) / newpos4.at<double>(2, 0));
        float y4 = (float)(newpos4.at<double>(1, 0) / newpos4.at<double>(2, 0));

        // cv::rectangle(copy_image_color, Rect(Point(x4, y3), Point(x2, y1)), Scalar(0, 255, 0), 2, 8, 0);
        Box_yolo box_basic = {x4, x2, y3, y1};
        boxes_2d_cluster.push_back(box_basic);
      }
      if (Box.y < 0)
      {
        // 카메라 원점 xyz 좌표 (3,1)생성
        cv::Mat newpos5(this->transformMat_right * pos1); // 카메라 좌표로 변환한 것.
        cv::Mat newpos6(this->transformMat_right * pos2);
        cv::Mat newpos7(this->transformMat_right * pos3);
        cv::Mat newpos8(this->transformMat_right * pos4);

        float x1 = (float)(newpos5.at<double>(0, 0) / newpos5.at<double>(2, 0) + 640);
        float y1 = (float)(newpos5.at<double>(1, 0) / newpos5.at<double>(2, 0));

        float x2 = (float)(newpos6.at<double>(0, 0) / newpos6.at<double>(2, 0) + 640);
        float y2 = (float)(newpos6.at<double>(1, 0) / newpos6.at<double>(2, 0));

        float x3 = (float)(newpos7.at<double>(0, 0) / newpos7.at<double>(2, 0) + 640);
        float y3 = (float)(newpos7.at<double>(1, 0) / newpos7.at<double>(2, 0));

        float x4 = (float)(newpos8.at<double>(0, 0) / newpos8.at<double>(2, 0) + 640);
        float y4 = (float)(newpos8.at<double>(1, 0) / newpos8.at<double>(2, 0));

        // cv::rectangle(copy_image_color, Rect(Point(x8, y8), Point(x6, y6)), Scalar(0, 255, 0), 2, 8, 0);
        Box_yolo box_basic = {x4, x2, y4, y2};
        boxes_2d_cluster.push_back(box_basic);
      }
    }

    // std_msgs::msg::Float64MultiArray cone_msg; // to planning
    // sensor_msgs::msg::PointCloud2 pointcloud_msg; // to rviz
    pcl::PointCloud<pcl::PointXYZI>::Ptr coord_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    for (int i = 0; i < boxes_2d_cluster.size(); i++)
    {
      float max_iou = 0.0f;
      int class_id = -1;

      for (int j = 0; j < yolo_boxes.size(); j++)
      {
        float iou = get_iou(boxes_2d_cluster[i], yolo_boxes[j]);
        if (iou > max_iou)
        {
          max_iou = iou;
          class_id = yolo_boxes[j].color;
        }
      }

      if (max_iou > 0.3f)
      {
        pcl::PointXYZI point_intensity;

        float center_x = lidar_boxes[i].x; // 3차원 상대좌표에서 x,y 가져옴
        float center_y = lidar_boxes[i].y;

        // float data
        this->cone_msg.data.push_back(center_x);
        this->cone_msg.data.push_back(center_y);
        this->cone_msg.data.push_back(class_id);
        // this->cone_msg.data.push_back(-1000.0);

        // pcl data
        if (class_id == 1)
        {
          point_intensity.x = center_x;
          point_intensity.y = center_y;
          point_intensity.z = 0.;
          point_intensity.intensity = 0.9;
        }
        else if (class_id == 2)
        {
          point_intensity.x = center_x;
          point_intensity.y = center_y;
          point_intensity.z = 0.;
          point_intensity.intensity = 0.3;
        }
        coord_cloud->push_back(point_intensity);
      }
    }
    // 플래닝으로 가는 데이터
    this->cone_pub_->publish(this->cone_msg);

    // 포인트클라우드 퍼블리시
    coord_cloud->width = 1;
    coord_cloud->height = coord_cloud->points.size();
    pcl::toROSMsg(*coord_cloud, this->pointcloud_msg);
    this->pointcloud_msg.header.frame_id = "velodyne";
    // pointcloud_msg.stamp = node->now();
    this->LiDAR_pub_->publish(this->pointcloud_msg);

    //====== 클러스터 박스 초기화 ==========
    this->box_locker = 0;
    boxes.clear();
    //==================================
    //======== 욜로박스 초기화 ============
    this->yolo_locker = 0;
    boxes_yolo.clear();
    //==================================

    this->is_rec_image = false;
    this->is_rec_box = false;
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