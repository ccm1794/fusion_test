// https://github.com/williamhyin/lidar_to_camera/blob/master/src/project_lidar_to_camera.cpp 투영은 이거 참고함


// 이 코드는 브랜치
#include <rclcpp/rclcpp.hpp>
#include <mutex>
#include <memory>
#include <thread>
#include <pthread.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

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

#include <interfaces/msg/new_detection3_d_array.hpp>
#include <vision_msgs/msg/bounding_box3_d.hpp>
#include <vision_msgs/msg/detection3_d.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>


using namespace std;
using namespace cv;

static bool IS_IMAGE_CORRECTION = true;

std::mutex mut_img;
std::mutex mut_pc;
std::mutex mut_box;

// Yolo Global parameter
std::mutex mut_yolo;
std::string Class_name;

pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr copy_raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>);

// 카메라 이미지
cv::Mat image_color;
cv::Mat overlay;
cv::Mat copy_image_color;
cv::Mat img_HSV;

// 박스 투영을 위한 이미지
cv::Mat img_box;
cv::Mat box_overlay;

int locker = 0;
int obj_count;
int yolo_num[10][5];

int cluster_count;
int box_coord[15][12]; // 이게 아니라 벡터로 저장할까?

struct Box // 박스 좌표를 저장하기 위한 구조체
{
  float x;
  float y;
  float z;

  float size_x;
  float size_y;
  float size_z;
};

struct Box_points
{
  float x1,y1,z1;
  float x2,y2,z2;
  float x3,y3,z3;
  float x4,y4,z4;
};

bool compareBoxes(const Box& a, const Box& b) 
{
  return a.x < b.x;
}

Box_points calcBox_points(const Box &box)
{
  Box_points vertices;
  if(box.y >= 0)
  {
    vertices.x1 = box.x - (box.size_x)/2;
    vertices.y1 = box.y + (box.size_y)/2;
    vertices.z1 = box.z - (box.size_z)/2;

    vertices.x2 = box.x + (box.size_x)/2;
    vertices.y2 = box.y - (box.size_y)/2;
    vertices.z2 = box.z - (box.size_z)/2;

    vertices.x3 = box.x + (box.size_x)/2;
    vertices.y3 = box.y - (box.size_y)/2;
    vertices.z3 = box.z + (box.size_z)/2;

    vertices.x4 = box.x - (box.size_x)/2;
    vertices.y4 = box.y + (box.size_y)/2;
    vertices.z4 = box.z + (box.size_z)/2;
  }
  else if(box.y < 0)
  {
    vertices.x1 = box.x - (box.size_x)/2;
    vertices.y1 = box.y - (box.size_y)/2;
    vertices.z1 = box.z - (box.size_z)/2;

    vertices.x2 = box.x + (box.size_x)/2;
    vertices.y2 = box.y + (box.size_y)/2;
    vertices.z2 = box.z - (box.size_z)/2;

    vertices.x3 = box.x + (box.size_x)/2;
    vertices.y3 = box.y + (box.size_y)/2;
    vertices.z3 = box.z + (box.size_z)/2;

    vertices.x4 = box.x - (box.size_x)/2;
    vertices.y4 = box.y - (box.size_y)/2;
    vertices.z4 = box.z + (box.size_z)/2;
  }
  return vertices;
}

std::vector<Box> boxes;

bool is_rec_image = false;
bool is_rec_LiDAR = false;

class ImageLiDARFusion : public rclcpp::Node
{
public:
  vector<double> CameraExtrinsic_vector;
  vector<double> CameraMat_vector;
  vector<double> DistCoeff_vector;
  vector<int> ImageSize_vector;

  Mat concatedMat;
  Mat transformMat;
  Mat CameraExtrinsicMat;
  Mat CameraMat;
  Mat DistCoeff;
  Mat frame1;

  pthread_t tids1_; //스레드 아이디

  int img_height = 480; // 왜 int는 못불러 오는가...
  int img_width = 640;

  float maxlen =200.0;         /**< Max distance: LiDAR */
  float minlen = 0.0001;        /**< Min distance: LiDAR */
  float max_FOV = CV_PI/4;     /**< Max FOV : Camera */

  sensor_msgs::msg::PointCloud2 colored_msg;

public:
  ImageLiDARFusion()
  : Node("projection_box")
  {
    RCLCPP_INFO(this->get_logger(), "------------ intialize ------------\n");

    this->declare_parameter("CameraExtrinsicMat", vector<double>());
    this->CameraExtrinsic_vector = this->get_parameter("CameraExtrinsicMat").as_double_array();
    this->declare_parameter("CameraMat", vector<double>());
    this->CameraMat_vector = this->get_parameter("CameraMat").as_double_array();
    this->declare_parameter("DistCoeff", vector<double>());
    this->DistCoeff_vector = this->get_parameter("DistCoeff").as_double_array();
    // this->declare_parameter("ImageSize", vector<int>());
    // this->ImageSize_vector = this->get_parameter("ImagSize").get_value<vector<int>>();
    this->set_param();

    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("corrected_image", 10);
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
       "/video1", 100,
       [this](const sensor_msgs::msg::Image::SharedPtr msg) -> void
       {
         ImageCallback(msg);
       }); //람다함수를 사용했는데 왜?, 그리고 일반적인 방법으로 하면 동작하지 않는다. 이유를 모르겠다

    LiDAR_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("test_LiDAR", 10);
    LiDAR_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/velodyne_points_filtered_center", 100,
      [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) -> void
      {
        LiDARCallback(msg);
      });
    
    yolo_count_sub_ = this->create_subscription<std_msgs::msg::Int16>(
      "/yolo_count", 10,
      [this](const std_msgs::msg::Int16::SharedPtr msg) -> void
      {
        YOLOCountCallback(msg);
      });
    
    yolo_detect_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
      "/yolo_detect", 10,
      [this](const vision_msgs::msg::Detection2DArray::SharedPtr msg) -> void
      {
        YOLOCallback(msg);
      });

    lidar_box_sub_ = this->create_subscription<interfaces::msg::NewDetection3DArray>(
      "/lidar_bbox", 10,
      [this](const interfaces::msg::NewDetection3DArray::SharedPtr msg) -> void
      {
        BoxCallback(msg);
      });
    
    int ret1 = pthread_create(&this->tids1_, NULL, publish_thread, this);
    RCLCPP_INFO(this->get_logger(), "------------ intialize end------------\n");

  }

  ~ImageLiDARFusion()
  {
    pthread_join(this->tids1_, NULL);
  }

public:
  void set_param();
  void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  void LiDARCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void YOLOCountCallback(const std_msgs::msg::Int16::SharedPtr msg);
  void YOLOCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);
  void BoxCallback(const interfaces::msg::NewDetection3DArray::SharedPtr msg);

  static void * publish_thread(void * this_sub);

private:
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr LiDAR_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr LiDAR_sub_;
  //yolo에서 (object 카운트), (클래스 이름, 박스 좌표) 받아오는 것 추가하기 
  rclcpp::Subscription<std_msgs::msg::Int16>::SharedPtr yolo_count_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr yolo_detect_sub_;
  rclcpp::Subscription<interfaces::msg::NewDetection3DArray>::SharedPtr lidar_box_sub_;
};


void ImageLiDARFusion::BoxCallback(const interfaces::msg::NewDetection3DArray::SharedPtr msg)
{
  cluster_count = msg->len.data;
  cout << "cluster count : " << cluster_count << endl;
  // 메세지가 얼마나 빠르게 들어오나 확인용 : velodyne_cluster.cpp에서 쏠 때 시간과 큰 차이가 없음.(0.02초)
  // RCLCPP_INFO(this->get_logger(), "cluster size : %d", cluster_count);
  if (cluster_count != 0 && locker == 0)
  {
    mut_box.lock();
    locker =1;
    for(int i = 0; i < cluster_count; i++)
    {
      Box box =
      {
        msg->detections[i].bbox.center.position.x, 
        msg->detections[i].bbox.center.position.y, 
        msg->detections[i].bbox.center.position.z, 
        msg->detections[i].bbox.size.x, 
        msg->detections[i].bbox.size.y, 
        msg->detections[i].bbox.size.z
      };
      //cout << "반복문 내부 : " << i << endl;
      boxes.push_back(box);

      // cout << "x : " << boxes[i].x << endl;
      // cout << "y : " << boxes[i].y << endl;
      // cout << "z : " << boxes[i].z << endl;
      // cout << "size_x : " << boxes[i].size_x << endl;
      // cout << "size_y : " << boxes[i].size_y << endl;
      // cout << "size_z : " << boxes[i].size_z << endl;
      // cout << "==========================" << endl;
    }
    std::sort(boxes.begin(), boxes.end(), compareBoxes); // 순서대로 나열해주는 코드
    int k = 0;
    for (const auto &Box : boxes)
    {
      //Box_points vertices = calcBox_points(Box);
      cout << "도나? : " << k << endl;
      k++;
    }
    mut_box.unlock();
    // boxes.clear();
  }
}

void ImageLiDARFusion::set_param()
{
  // 받아온 파라미터
  Mat CameraExtrinsicMat_(4, 4, CV_64F, CameraExtrinsic_vector.data());
  Mat CameraMat_(3, 3, CV_64F, CameraMat_vector.data()); 
  Mat DistCoeffMat_(1, 4, CV_64F, DistCoeff_vector.data());

  CameraExtrinsicMat_.copyTo(this->CameraExtrinsicMat);
  CameraMat_.copyTo(this->CameraMat);
  DistCoeffMat_.copyTo(this->DistCoeff);
  

  //재가공 : 회전변환행렬, 평행이동행렬
  Mat Rlc = CameraExtrinsicMat(cv::Rect(0,0,3,3));
  Mat Tlc = CameraExtrinsicMat(cv::Rect(3,0,1,3));

  cout << "Tlc : " << Tlc << "\n";

  cv::hconcat(Rlc, Tlc, this->concatedMat);

  cout << "concatedMat : " << this->concatedMat << "\n";

  //재가공 : transformMat
  this->transformMat = this->CameraMat * this->concatedMat;

  //재가공 : image_size
  // this->img_height = this->ImageSize_vector[0];
  // this->img_width = this->ImageSize_vector[1];
  RCLCPP_INFO(this->get_logger(), "ok");
}

void ImageLiDARFusion::ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // rclcpp::WallRate loop_rate(20.0);
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

  //Mat image(msg->height, msg->width, CV_8UC3, const_cast<unsigned char*>(msg->data.data()), msg->step);

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
    //image_color = cv_ptr->image.clone();
    image_color = image.clone();
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

  image_pub_->publish(std::move(image_msg));
  // imshow("image", image_color);
  // waitKey(1);
}

void ImageLiDARFusion::LiDARCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  mut_pc.lock();
  pcl::fromROSMsg(*msg, *raw_pcl_ptr);
  mut_pc.unlock();
  is_rec_LiDAR = true;
}

void ImageLiDARFusion::YOLOCountCallback(const std_msgs::msg::Int16::SharedPtr msg)
{
  obj_count = msg->data;
}

void ImageLiDARFusion::YOLOCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
  if (obj_count != 0)
  {
    std::string Class_name;

    mut_yolo.lock();

    for(int i = 0; i<obj_count; i++)
    {
      // cout << msg->detections[i].results.size() << endl;

      Class_name = msg->detections[i].results[0].id;
      // cout << Class_name << endl;

      int color = 0;
      if(Class_name == "blue")
      {
        color = 1;
      }
      else if(Class_name == "orange")
      {
        color = 2;
      }
      else
      {
        color = 0;
      }

      yolo_num[i][0] = color;
      yolo_num[i][1] = msg->detections[i].bbox.center.x - (msg->detections[i].bbox.size_x)/2;
      yolo_num[i][2] = msg->detections[i].bbox.center.x + (msg->detections[i].bbox.size_x)/2;
      yolo_num[i][3] = msg->detections[i].bbox.center.y - (msg->detections[i].bbox.size_y)/2;
      yolo_num[i][4] = msg->detections[i].bbox.center.y + (msg->detections[i].bbox.size_y)/2;
      
    }
    // cout << "-----------------" << endl;
    mut_yolo.unlock();
  }
  else
  {
  }
}

void * ImageLiDARFusion::publish_thread(void * args)
{
  ImageLiDARFusion * this_sub = (ImageLiDARFusion *)args;
  rclcpp::WallRate loop_rate(10.0);
  while(rclcpp::ok())
  {
    if (is_rec_image && is_rec_LiDAR)
    {
      
      //**point** : Syncronize topics : PointCloud, Image, Yolomsg 싱크 맞추는게 중요!
      mut_pc.lock();
      pcl::copyPointCloud(*raw_pcl_ptr, *copy_raw_pcl_ptr);
      mut_pc.unlock();

      mut_img.lock();
      //copy_image_color = image_color.clone();
      overlay = image_color.clone();
      mut_img.unlock();

      // mut_yolo.lock();
      // copy_img_Yolo = img_Yolo;
      // mut_yolo.unlock();

      mut_box.lock();
      box_overlay = image_color.clone();
      mut_box.unlock();

      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_xyzinten(new pcl::PointCloud<pcl::PointXYZI>);
      const int size = copy_raw_pcl_ptr->points.size();

      // cout << "point cloud size : " << size << endl;

      Mat image_show = image_color.clone();

      for (int i = 0; i < size ; i++) //pointclouds 투영
      {
        pcl::PointXYZI pointColor;
        
        // **앵글이 카메라 수평 화각 안에 들어오는가?**
        pcl::PointXYZ temp_;
        temp_.x = copy_raw_pcl_ptr->points[i].x;
        temp_.y = copy_raw_pcl_ptr->points[i].y;
        temp_.z = copy_raw_pcl_ptr->points[i].z;           

        float R_ = sqrt(pow(temp_.x,2) + pow(temp_.y,2)); // 라이다로부터 수평거리
        float azimuth_ = abs(atan2(temp_.y, temp_.x)); // 방위각
        float elevation_ = abs(atan2(R_, temp_.z)); // 고도각

        pointColor.x = copy_raw_pcl_ptr->points[i].x;
        pointColor.y = copy_raw_pcl_ptr->points[i].y;
        pointColor.z = copy_raw_pcl_ptr->points[i].z;

        if(azimuth_ > (this_sub->max_FOV))
        {
          // pointColor.intensity=0.;
          continue; //처리하지 않겠다
        }
        else if(azimuth_ <= (this_sub->max_FOV))
        {
          //색깔점에 좌표 대입
          
          //라이다 좌표 행렬(4,1)
          double a_[4] = {pointColor.x, pointColor.y, pointColor.z, 1.0};
          cv::Mat pos( 4, 1, CV_64F, a_); // 라이다 좌표

          //카메라 원점 xyz 좌표 (3,1)생성
          cv::Mat newpos(this_sub->transformMat * pos); // 카메라 좌표로 변환한 것.

          //카메라 좌표(x,y) 생성
          float x = (float)(newpos.at<double>(0, 0) / newpos.at<double>(2, 0));
          float y = (float)(newpos.at<double>(1, 0) / newpos.at<double>(2, 0));

          // trims viewport according to image size
          float dist_ = sqrt(pow(pointColor.x,2) + pow(pointColor.y,2) + pow(pointColor.z,2));

          if (this_sub->minlen < dist_ && dist_ < this_sub->maxlen)
          {
            if (x >= 0 && x < this_sub->img_width && y >= 0 && y < this_sub->img_height)
            {
              // cout << "2" << endl; // 여기서부터 코드가 안돈다.->내부 파라미터가 올바르지 않아 그랬음
              // imread BGR (BITMAP);
              int row = int(y);
              int column = int(x);
              pointColor.intensity = 0.9;

              cv::Point pt;
              pt.x = x;
              pt.y = y;

              float val = pointColor.x; // 라이다 좌표에서 x를 뜻함
              float maxVal = 100.0;

              int green = min(255, (int) (255 * abs((val - maxVal) / maxVal)));
              int red = min(255, (int) (255 * (1 - abs((val - maxVal) / maxVal))));
              cv::circle(overlay, pt, 5, cv::Scalar(0, green, red), -1);

              // 욜로 박스치는 부분
              // for(int j = 0; j < obj_count; j++)
              // {
              //   // cout << "3" << endl;
              //   if((column >= yolo_num[j][1]) && (column <= yolo_num[j][2]))
              //   {
              //     // cout << "4" << endl;
              //     if ((row >= yolo_num[j][3]) && (row <= yolo_num[j][4]))
              //     {
              //       // cerr << "Lavacon type is " << yolo_num[j][0] << endl;
              //       // pointColor.intensity = yolo_num[j][0];
              //       if(yolo_num[j][0] == 1)
              //       {
              //         pointColor.intensity = 0.7;
              //       }
              //       else if(yolo_num[j][0] == 2)
              //       {
              //         pointColor.intensity = 0.3;
              //       }
              //       break; 
              //     }
              //   }
              //   else
              //   {
              //     // cerr << "Lavacon type is 0" << endl;
              //   }
              // }
            }
          }
        }
      pc_xyzinten->push_back(pointColor);
      }


//======================================================================================
      //for (const auto& Box : boxes) //boxes의 좌표 투영 - for문에 진입하지 못한다.
      // for(int i = 0; i < cluster_count; i++)
      for (const auto& Box : boxes)
      {
        
        Box_points vertices = calcBox_points(Box);
        double box_1[4] = {vertices.x1, vertices.y1, vertices.z1, 1.0};
        double box_2[4] = {vertices.x2, vertices.y2, vertices.z2, 1.0};
        double box_3[4] = {vertices.x3, vertices.y3, vertices.z3, 1.0};
        double box_4[4] = {vertices.x4, vertices.y4, vertices.z4, 1.0};


        cv::Mat pos1( 4, 1, CV_64F, box_1); // 3차원 좌표
        cv::Mat pos2( 4, 1, CV_64F, box_2);
        cv::Mat pos3( 4, 1, CV_64F, box_3);
        cv::Mat pos4( 4, 1, CV_64F, box_4);

        //카메라 원점 xyz 좌표 (3,1)생성
        cv::Mat newpos1(this_sub->transformMat * pos1); // 카메라 좌표로 변환한 것.
        cv::Mat newpos2(this_sub->transformMat * pos2);
        cv::Mat newpos3(this_sub->transformMat * pos3);
        cv::Mat newpos4(this_sub->transformMat * pos4);

        //카메라 좌표(x,y) 생성
        float x1 = (float)(newpos1.at<double>(0, 0) / newpos1.at<double>(2, 0));
        float y1 = (float)(newpos1.at<double>(1, 0) / newpos1.at<double>(2, 0));

        float x2 = (float)(newpos2.at<double>(0, 0) / newpos2.at<double>(2, 0));
        float y2 = (float)(newpos2.at<double>(1, 0) / newpos2.at<double>(2, 0));

        float x3 = (float)(newpos3.at<double>(0, 0) / newpos3.at<double>(2, 0));
        float y3 = (float)(newpos3.at<double>(1, 0) / newpos3.at<double>(2, 0));

        float x4 = (float)(newpos4.at<double>(0, 0) / newpos4.at<double>(2, 0));
        float y4 = (float)(newpos4.at<double>(1, 0) / newpos4.at<double>(2, 0));

        // cout << "x1 : " << x1 << " y1 : " << y1 << endl;
        std::vector<cv::Point> projected_boxes =
        {
          cv::Point(x1, y1),
          cv::Point(x2, y2),
          cv::Point(x3, y3),
          cv::Point(x4, y4)
        };
        // draw polygon
        cv::polylines(box_overlay, projected_boxes, true, cv::Scalar(0, 255, 0), 2);
      }
      
      boxes.clear();
      locker = 0;
//========================================================================================
      float opacity = 0.6;
      cv::addWeighted(overlay, opacity, image_color, 1 - opacity, 0, image_color);
      cv::addWeighted(box_overlay, opacity, image_color, 1 - opacity, 0, image_color);

      string windowName = "LiDAR data on image overlay";
      cv::namedWindow(windowName, 3);
      cv::imshow(windowName, image_color);
      char ch = cv::waitKey(10);
      if(ch == 27) break;

      for(int i = 0; i < 5; i++)
      {
        for(int j = 0; j < 10; j++)
        {
          yolo_num[i][j] = 0;
        }
      }
      for(int i = 0; i < 12; i++)
      {
        for(int j = 0; j < 15; j++)
        {
          box_coord[i][j] = 0;
        }
      }
      pc_xyzinten->width = 1;
      pc_xyzinten->height = pc_xyzinten->points.size();
      pcl::toROSMsg(*pc_xyzinten, this_sub->colored_msg);
      this_sub->colored_msg.header.frame_id = "velodyne";
      this_sub->colored_msg.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
      this_sub->LiDAR_pub_->publish(this_sub->colored_msg);
      loop_rate.sleep();

    }
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