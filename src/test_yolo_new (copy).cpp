#include <rclcpp/rclcpp.hpp>
#include <mutex>
#include <memory>
#include <functional>
#include <string>

#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud_conversion.hpp>
#include <sensor_msgs/image_encodings.hpp>
//#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/msg/point.hpp>
#include <image_transport/image_transport.h> //꼭 있을 필요는 없을 듯?
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// 아래 라이브러리는 yolov7에 맞게 바꿔줘야함.
// #include <darknet_ros_msgs/BoundingBox.h> //이거는 안쓰는거 같다.
// #include <darknet_ros_msgs/BoundingBoxes.h>
// #include <darknet_ros_msgs/ObjectCount.h>

// ///////////////////////////////////////////////////////////////////////
// topic: /test_lidar
// frame: sensor_frame
// multi-topic asynchronized //비동기화
// ///////////////////////////////////////////////////////////////////////

using namespace std;
using namespace cv;
typedef pcl::PointXYZRGB PointType; //안쓰는거 같다.

static bool IS_IMAGE_CORRECTION = true;

std::mutex mut_img;
std::mutex mut_pc;

//포인트클라우드 메세지 : XYZI
pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr copy_raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>);

//카메라 이미지
cv::Mat image_color;
cv::Mat copy_image_color;
cv::Mat img_HSV;

//욜로 글로벌 파라미터
std::mutex mut_yolo;
std::string Class_name;
int obj_count;
int yolo_num[10][5]; //2차원 배열. 객체를 10개까지 담겠다는 건가?

//이미지, 라이다 들어오는지 확인
bool is_rec_image = false;
bool is_rec_LiDAR = false;

pthread_t  tids1_;

cv::Mat transform_matrix;    /**< from globol to image coordinate */
cv::Mat intrinsic_matrix;    /**< from local to image coordinate  */
cv::Mat extrinsic_matrix;    /**< from global to local coordinate */
cv::Mat dist_matrix;         /**< dist parameters  */
cv::Mat frame1;              /**< Camera image */

int img_height;
int img_width;

float maxlen =200.0;         /**< Max distance: LiDAR */
float minlen = 0.001;        /**< Min distance: LiDAR */
float max_FOV = CV_PI/4;     /**< Max FOV : Camera */

cv_bridge::CvImagePtr cv_ptr;
//시작
class ImageLiDARFusion : public rclcpp::Node
{
public:
  ImageLiDARFusion()
  : Node("fusion")
  {
    RCLCPP_INFO(this->get_logger(), "------------ intialize ------------\n");
    //카메라 파라미터 가져오기
    this->declare_parameter("CameraExtrinsicMat", vector<double>());
    vector<double> extrinsic_param = this->get_parameter("CameraExtrinsicMat").as_double_array();
    this->declare_parameter("CameraMat", vector<double>());
    vector<double> intrinsic_param = this->get_parameter("CameraMat").as_double_array();
    this->declare_parameter("DistCoeff", vector<double>());
    vector<double> dist_param = this->get_parameter("DistCoeff").as_double_array();
    this->declare_parameter("ImageSize", vector<int>());
    vector<int> img_size = this->get_parameter("ImageSize").as_integer_array();
    //set_param으로 가져가서 형식에 맞게 설정
    set_param();

    //퍼블리셔, 서브스크라이버 선언
    //이미지 펍,섭
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("corrected_camera", 10);
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera_fusion_test", 100, std::bind(&ImageLiDARFusion::ImageCallback, this, _1));
    //라이다 펍,섭
    LiDAR_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("test_LiDAR", 10);
    LiDAR_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/velodyne_points", 100, std::bind(&ImageLiDARFusion::LiDARCallback, this, _1));
    //욜로 섭 - 이건 나중에 바꿔야함.
    detect_obj_ = this->create_subscription<darknet_ros_msgs::msg::BoundingBoxes>(
      "/darknet_ros/bounding_boxes", 10, std::bind(&ImageLiDARFusion::YoloCallback, this, _1));
    obj_num_ = this->create_subscription<darknet_ros_msgs::msg::ObjectCount>(
      "/darknet_ros/found_object", 10, std::bind(&ImageLiDARFusion::ObjectCount, this, _1));

    //새로운 쓰레드 create
    int ret1 = pthread_create(&tids1_, NULL,  publish_thread, this);
    //publish_thread();

    RCLCPP_INFO(this->get_logger(), "START LISTENING\n");
  };

  ~ImageLiDARFusion()
  {
  };

  static void * publish_thread(void * this_sub);

private:
  void set_param()
  {
    //yaml파일의 파라미터를 위에 선언한 Mat에 넣어준다.

    //extrinsic_matrix
    //cv::Mat extrinsic_mat_sub(4, 4, CV_64F, extrinsic_param.data()); //이 방법이 된다면 이게 더 나을듯
    cv::Mat concated_matrix;
    cv::Mat extrinsic_mat_sub(extrinsic_param);
    extrinsic_mat_sub = extrinsic_mat_sub.reshape(4, 4);
    cv::Mat Rlc = extrinsic_matrix(cv::Rect(0,0,3,3));
    cv::Mat Tlc = extrinsic_matrix(cv::Rect(3,0,1,3));
    cv::hconcat(Rlc, Tlc, this->concated_matrix);

    //intrinsic_matrix
    intrinsic_matrix(intrinsic_param);
    intrinsic_matrix = intrinsic_matrix.reshape(3,3);

    //transform_matrix
    this->transform_matrix = intrinsic_matrix * this->concated_matrix;

    //dist_matrix
    dist_matrix(dist_param);
    dist_matrix = dist_matrix.reshape(1,5);

    //img_size
    img_height = img_size[0];
    img_width = img_size[1];
  };

private:
  void LiDARCallback(const sensor_msgs::msg::PointCloud2::SharedPtr & msg)
  {
    mut_pc.lock();
    pcl::fromROSMsg(*msg, *raw_pcl_ptr);
    mut_pc.unlock();
    is_rec_LiDAR = true;
  };

private:
  void ObjectCount(darknet_ros_msgs::msg::ObjectCount ObjectCount)
  {
    obj_count = ObjectCount.count;
  };

private:
  void YoloCallback(darknet_ros_msgs::BoundingBoxes boundingBox)
  {
    if(obj_count != 0)
    {
      std::string Class_name;

      mut_yolo.lock();

      for(int i=0; i<obj_count; i++)
      {
        Class_name = boundingBox.bounding_boxes[i].Class;

        int color = 0;
        if(Class_name == "BlueCone"){color = 1;}
        else if(Class_name == "RedCone"){color = 2;}
        else{color = 0;}

        yolo_num[i][0] = color;
        yolo_num[i][1] = boundingBox.bounding_boxes[i].xmin;
        yolo_num[i][2] = boundingBox.bounding_boxes[i].xmax;
        yolo_num[i][3] = boundingBox.bounding_boxes[i].ymin;
        yolo_num[i][4] = boundingBox.bounding_boxes[i].ymax;
      }

      mut_yolo.unlock();
    }
    else{}
  };

private:
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr & msg)
  {
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch(const std::exception& e)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
    // 왜곡보정
    if(IS_IMAGE_CORRECTION)
    {
      mut_img.lock();
      cv::undistort(cv_ptr->image, image_color, this->intrinsic_matrix, this->dist_matrix);
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
    sensor_msgs::msg::Image::SharedPtr linshi_msg = cv_ptr->toImageMsg();
    linshi_msg->header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
    image_pub_->publish(linshi_msg);

    frame1 = cv_bridge::toCvShare(msg, "bgr8")->image;
  };

};

void * ImageLiDARFusion::publish_thread(void * args)
{
  ImageLiDARFusion * this_sub = (ImageLiDARFusion *) args;
  rclcpp::WallRate loop_rate(20);
  while(rclcpp::ok())
  {
    if(is_rec_image && is_rec_LiDAR)
    {
      //**point** : Syncronize topics : PointCloud, Image, Yolomsg
      mut_pc.lock();
      pcl::copyPointCloud(*raw_pcl_ptr, *copy_raw_pcl_ptr);
      mut_pc.unlock();

      mut_img.lock();
      copy_image_color = image_color.clone();
      cvtColor(copy_image_color, img_HSV, COLOR_BGR2HSV);
      mut_img.unlock();

      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_xyzinten(new pcl::PointCloud<pcl::PointXYZI>);
      const int size = copy_raw_pcl_ptr->points.size();

      for(int i =0; i<size; i++)
      {
        pcl::PointXYZI pointColor;

        pcl::PointXYZ temp_;
        temp_.x = copy_raw_pcl_ptr->points[i].x;
        temp_.y = copy_raw_pcl_ptr->points[i].y;
        temp_.z = copy_raw_pcl_ptr->points[i].z;
        float R_ = sqrt(pow(temp_.x,2)+pow(temp_.y,2)) //+pow(temp_.z,2)); R_ = 거리
        float azimuth_ = abs(atan2(temp_.y, temp_.x)); //azimuth = 방위각
        float elevation_ = abs(atan2(R_, temp_.z)); //elevation = 고도각

        pointColor.x = copy_raw_pcl_ptr->points[i].x;
        pointColor.y = copy_raw_pcl_ptr->points[i].y;
        pointColor.z = copy_raw_pcl_ptr->points[i].z;

        if(azimuth_ > (this_sub->max_FOV))
        {
          pointColor.intensity = 0;
        }
        else if(azimuth_ <= (this_sub->max_FOV))
        {
          // 색깔점에 그 점의 xyz 좌표 대입
          // 라이다 좌표 행렬 (4,1) 생성
          double a_[4] = {pointColor.x, pointColor.y, pointColor.z, 1.0};
          cv::Mat pos(4,1,CV_64F, a_); //라이다 좌표 행렬 선언

          //카메라 원점 xyz좌표 (3,1) 생성
          cv::Mat newpos(this_sub->transform_matrix * pos);

          //카메라 좌표(x, y) 생성
          float x = (float)(newpos.at<double>(0, 0) / newpos.at<double>(2, 0));
          float y = (float)(newpos.at<double>(1, 0) / newpos.at<double>(2, 0));

          //카메라 좌표(x, y)가 이미지 크기 안에 있을 때
          float dist_ = sqrt(pow(pointColor.x,2) + pow(pointColor.y,2) + pow(pointColor.z,2));

          if (this_sub->minlen < dist_ && dist_ < this_sub->maxlen)
          {
            if(x >= 0 && x < this_sub->img_width && y >= 0 && y < this_sub->img_height)
            {
              //imread BGR (BITMAP)
              int row = int(y);
              int column = int(y);

              for(int j = 0; j < obj_count; j++)
              {
                if((column >= yolo_num[j][1]) && (column <= yolo_num[j][2]))
                {
                  if((row >= yolo_num[j][3]) && (row <= yolo_num[j][4]))
                  {
                    cerr << "Lavacon type is " << yolo_num[j][0] << endl;
                    pointColor.intensity = yolo_num[j][0];
                    break;
                  }
                }
                else
                {
                  cerr << "Lavacon type is 0" << endl;
                }
              }

            }
          }
        }
      pc_xyzinten->push_back(pointColor);  
      }  
      for(int i = 0; i<5; i++)
      {
        for(int j = 0; j<10; j++)
        {
          yolo_num[i][j] = 0;
        }
      }
      pc_xyzinten->width = 1;
      pc_xyzinten->height = pc_xyzinten->points.size();
      pcl::toROSMsg(*pc_xyzinten, this_sub->colored_msg);
      this_sub->colored_msg.header.frame_id = "velodyne";
      this_sub->colored_msg.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
      this_sub->LiDAR_pub->publish(this_sub->colored_msg);
      loop_rate.sleep();
    }  
  }
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  if(argc != 2)
  {
    RCLCPP_ERROR(this->get_logger(), "Please input yaml file name");
    return -1;
  }
  if(strcmp(argv[1], "true") == 0)
  {
    IS_IMAGE_CORRECTION = true;
    RCLCPP_INFO(this->get_logger(), "Image Correction ON");
  }
  else
  {
    IS_IMAGE_CORRECTION = false;
    RCLCPP_INFO(this->get_logger(), "Image Correction OFF");
  }

  auto node = std::make_shared<ImageLiDARFusion>();
  if (node == nullptr) {
    RCLCPP_ERROR(node->get_logger(), "Failed to create node");
    return 1;
  }
  auto executor = rclcpp::executors::MultiThreadedExecutor(4); //쓰레드 개수
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}