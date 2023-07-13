#include <ros/ros.h>
#include <mutex>

#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <darknet_ros_msgs/ObjectCount.h>

// ///////////////////////////////////////////////////////////////////////
// topic: /test_lidar
// frame: sensor_frame
// multi-topic asynchronized
// ///////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;
typedef pcl::PointXYZRGB PointType; // 색깔점

static bool IS_IMAGE_CORRECTION = true;

std::mutex mut_img;
std::mutex mut_pc;



//포인트클라우드 메시지 : XYZI
pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>); 
pcl::PointCloud<pcl::PointXYZI>::Ptr copy_raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>); 

// 카메라 이미지
cv::Mat image_color;
cv::Mat copy_image_color;
cv::Mat img_HSV;

// Yolo Global parameter
std::mutex mut_yolo;
std::string Class_name;
int obj_count;
int yolo_num[10][5];

// 이미지, 라이다 들어오는지 확인
bool is_rec_image = false;
bool is_rec_LiDAR = false;



class ImageLiDARFusion
{
public:
  ros::NodeHandle nh;
  image_transport::ImageTransport it;
  image_transport::Subscriber image_sub;
  ros::Subscriber LiDAR_sub;
  image_transport::Publisher image_pub;
  ros::Publisher LiDAR_pub;
  cv_bridge::CvImagePtr cv_ptr;
  
  sensor_msgs::PointCloud out_pointcloud;
  sensor_msgs::PointCloud2 colored_msg; 

  // YOLO topic
  ros::Subscriber detect_obj;
  ros::Subscriber obj_num; 
      
  pthread_t  tids1_;

public:
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

public:
  ImageLiDARFusion():it(nh)
  {
    ROS_INFO("------------ intialize ----------\n");
    this->set_param();

    // image_sub = it.subscribe("/camera/stopline/image_raw", 100, &ImageLiDARFusion::imageCallback, this);
    image_sub = it.subscribe("/camera_fusion_test", 100, &ImageLiDARFusion::imageCallback, this);

    // LiDAR_sub = nh.subscribe("/velodyne_cluster", 100, &ImageLiDARFusion::LiDARCallback, this);
    // LiDAR_sub = nh.subscribe("/velodyne_points", 100, &ImageLiDARFusion::LiDARCallback, this);
    LiDAR_sub = nh.subscribe("/coord/centerP_to_test01", 100, &ImageLiDARFusion::LiDARCallback, this);

    detect_obj = nh.subscribe<darknet_ros_msgs::BoundingBoxes>("/darknet_ros/bounding_boxes", 1000, &ImageLiDARFusion::YoloCallback,this);
    obj_num = nh.subscribe<darknet_ros_msgs::ObjectCount>("/darknet_ros/found_object", 1000, &ImageLiDARFusion::ObjectCount,this);
    
    image_pub = it.advertise("/corrected_camera", 10);    
    LiDAR_pub = nh.advertise<sensor_msgs::PointCloud2>("test_LiDAR", 10);


    int ret1 = pthread_create(&tids1_, NULL,  publish_thread, this); 

    ROS_INFO("START LISTENING\n");
  };

  ~ImageLiDARFusion()
  {
  };

  void set_param();
  void LiDARCallback(const sensor_msgs::PointCloud2ConstPtr & msg);
  void imageCallback(const sensor_msgs::ImageConstPtr& msg);
  void ObjectCount(darknet_ros_msgs::ObjectCount ObjectCount);
  void YoloCallback(darknet_ros_msgs::BoundingBoxes msg);


  static void * publish_thread(void *  this_sub);
};



void * ImageLiDARFusion::publish_thread(void * args)
{
  ImageLiDARFusion * this_sub = (ImageLiDARFusion *) args; //**point3**：입력된 void*매개변수를 클래스 객체(ImageLiDARFusion*)로 반환 바늘?(???)
  ros::Rate loop_rate(10);
  while (ros::ok())
  {
  
    if (is_rec_image && is_rec_LiDAR)
    {
      //**point** : Syncronize topics : PointCloud, Image, Yolomsg
      mut_pc.lock();
      pcl::copyPointCloud (*raw_pcl_ptr, *copy_raw_pcl_ptr);
      mut_pc.unlock();

      mut_img.lock();
      copy_image_color = image_color.clone();
      cvtColor(copy_image_color,img_HSV,COLOR_BGR2HSV);
      mut_img.unlock();

      // mut_yolo.lock();
      // copy_img_Yolo = img_Yolo;
      // mut_yolo.unlock();

      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_xyzinten(new pcl::PointCloud<pcl::PointXYZI>);
      const int size = copy_raw_pcl_ptr->points.size(); 
      
      // cout<<"point cloud size: "<<size<< endl;;
      

      for (int i = 0; i < size; i++)
      {
        // project get the photo coordinate
        // pcl::PointXYZRGB pointRGB;
        pcl::PointXYZI pointColor;
        
        // ###Is angle in camera FOV range?###
        // PointType temp_;
        pcl::PointXYZ temp_;
        temp_.x = copy_raw_pcl_ptr->points[i].x;
        temp_.y = copy_raw_pcl_ptr->points[i].y;
        temp_.z = copy_raw_pcl_ptr->points[i].z;
        float R_ = sqrt(pow(temp_.x,2)+pow(temp_.y,2));

        // float azimuth_ = abs(atan2(temp_.x, temp_.y)); // output : in radian!!
        float azimuth_ = abs(atan2(temp_.y, temp_.x)); // output : in radian!!
        float elevation_ = abs(atan2(R_, temp_.z)); // 이걸 계산을 이걸로 하는게 맞나? 

        pointColor.x = copy_raw_pcl_ptr->points[i].x;
        pointColor.y = copy_raw_pcl_ptr->points[i].y;
        pointColor.z = copy_raw_pcl_ptr->points[i].z;

        if(azimuth_ > (this_sub->max_FOV) )
        {
          // cerr << "? ";
          pointColor.intensity=0;
          // continue;
        }
        else if (azimuth_ <= (this_sub->max_FOV))
        {
          // ###색깔점에 점의 x,y,z 대입
          
          //라이다 좌표 행렬(4,1) 생성
          double a_[4] = { pointColor.x, pointColor.y, pointColor.z, 1.0 };
          cv::Mat pos(4, 1, CV_64F, a_);

          //카메라 원점 xyz좌표 (3,1)생성
          cv::Mat newpos(this_sub->transform_matrix * pos);

          //카메라 좌표(x,y) 생성
          float x = (float)(newpos.at<double>(0, 0) / newpos.at<double>(2, 0));
          float y = (float)(newpos.at<double>(1, 0) / newpos.at<double>(2, 0));
          

          // ###Trims viewport according to image size###
          float dist_ = sqrt(pow(pointColor.x,2) + pow(pointColor.y,2) + pow(pointColor.z,2));

          if (this_sub->minlen <  dist_ && dist_ < this_sub->maxlen)
          
          {
            if (x >= 0 && x < this_sub->img_width && y >= 0 && y < this_sub->img_height)
            {
              //  imread BGR（BITMAP）
              int row = int(y);
              int column = int(x);
              

              for(int j= 0; j < obj_count; j++)
              {
                if((column >= yolo_num[j][1])&&(column <= yolo_num[j][2]))
                {
                  if((row >= yolo_num[j][3])&&(row <= yolo_num[j][4]))
                  {
                    cerr << "Lavacon type is " << yolo_num[j][0] << endl;
                    pointColor.intensity = yolo_num[j][0];
                    break;
                  }
                }
                else
                {
                  cerr << "Lavacon type is 0" << endl;zz
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
      pcl::toROSMsg(*pc_xyzinten,  this_sub->colored_msg); 
      this_sub->colored_msg.header.frame_id = "velodyne";
      this_sub->colored_msg.header.stamp = ros::Time::now(); 
      this_sub->LiDAR_pub.publish(this_sub->colored_msg); 
      loop_rate.sleep();
    }
  }
}

void ImageLiDARFusion::set_param() 
{
  // extrinsic matrix parameters
  XmlRpc::XmlRpcValue param_list;
  std::vector<double> Extrin_matrix;
  if(!nh.getParam("/test_yolo/CameraExtrinsicMat/data", param_list))
      ROS_ERROR("Failed to get extrinsic parameter.");
  ROS_INFO("\n get extrinsic parameter:");
  for (size_t i = 0; i < param_list.size(); ++i) 
  {
      XmlRpc::XmlRpcValue tmp_value = param_list[i];
      if(tmp_value.getType() == XmlRpc::XmlRpcValue::TypeDouble)
      {    
        Extrin_matrix.push_back(double(tmp_value));
        ROS_INFO("PARAME SIZE = %f", double(tmp_value));
      }
  }
  // Intrinsic matrix parameters
  std::vector<double> Intrinsic;
  if(!nh.getParam("/test_yolo/CameraMat/data", param_list))
      ROS_ERROR("Failed to get intrinsic parameter.");
  ROS_INFO("\n get intrinsic parameter:");
  for (size_t i = 0; i < param_list.size(); ++i) 
  {
      XmlRpc::XmlRpcValue tmp_value = param_list[i];
      if(tmp_value.getType() == XmlRpc::XmlRpcValue::TypeDouble)
      {    
        Intrinsic.push_back(double(tmp_value));
        ROS_INFO("PARAME SIZE = %f", double(tmp_value));
      }
  }


  // 5 distortion parameters
  std::vector<double> dist;
  if(!nh.getParam("/test_yolo/DistCoeff/data", param_list))
      ROS_ERROR("Failed to get distortion parameter.");
  ROS_INFO("\n get distortion parameter:");
  for (size_t i = 0; i < param_list.size(); ++i) 
  {
      XmlRpc::XmlRpcValue tmp_value = param_list[i];
      if(tmp_value.getType() == XmlRpc::XmlRpcValue::TypeDouble)
      {    
        dist.push_back(double(tmp_value));
        ROS_INFO("PARAME SIZE = %f", double(tmp_value));
      }
  }
  
  // img size
  std::vector<int> img_size;
  if(!nh.getParam("/test_yolo/ImageSize", param_list))
      ROS_ERROR("Failed to get extrinsic parameter.");
  ROS_INFO("\n get image size:");
  for (size_t i = 0; i < param_list.size(); ++i) 
  {
      XmlRpc::XmlRpcValue tmp_value = param_list[i];
      if(tmp_value.getType() == XmlRpc::XmlRpcValue::TypeInt)
      {    
        img_size.push_back(int(tmp_value));
        ROS_INFO("PARAME SIZE = %d", int(tmp_value));
      }
  }
  img_width = img_size[0];
  img_height = img_size[1];
  // cerr << img_width << img_height << endl;
  // convert cv::Mat
  cv::Mat dist_array(5, 1, CV_64F, &dist[0]);
  this->dist_matrix = dist_array.clone();

  
  cv::Mat Int(3, 3, CV_64F, &Intrinsic[0]);
  this->intrinsic_matrix = Int.clone();

  cv::Mat ext_(4, 4, CV_64F, &Extrin_matrix[0]);
  cv::Mat Rlc = ext_(cv::Rect(0, 0, 3, 3));  
  cv::Mat Tlc = ext_(cv::Rect(3, 0, 1, 3));
  cv::hconcat(Rlc, Tlc, this->extrinsic_matrix);
  // transform matrix: from global coordinate to image coordinate
  this->transform_matrix = Int * this->extrinsic_matrix; 
}

void ImageLiDARFusion::LiDARCallback(const sensor_msgs::PointCloud2ConstPtr & msg)
{
  mut_pc.lock();
  pcl::fromROSMsg(*msg, *raw_pcl_ptr);	
  mut_pc.unlock();
  is_rec_LiDAR = true;
}

void ImageLiDARFusion::ObjectCount(darknet_ros_msgs::ObjectCount ObjectCount)
{
    obj_count = ObjectCount.count;
}

void ImageLiDARFusion::YoloCallback(darknet_ros_msgs::BoundingBoxes boundingBox)
{
  if(obj_count != 0)
  {

    std::string Class_name;
    
    mut_yolo.lock();
    
    for(int i=0; i<obj_count;i++)
    {
      Class_name = boundingBox.bounding_boxes[i].Class;

      int color = 0;
      if(Class_name == "BlueCone")
      {
        color = 1;
      }
      else if(Class_name == "RedCone")
      {
        color = 2;
      }
      else
      {
        color = 0;
      }

      yolo_num[i][0] = color;
      yolo_num[i][1] = boundingBox.bounding_boxes[i].xmin;
      yolo_num[i][2] = boundingBox.bounding_boxes[i].xmax;
      yolo_num[i][3] = boundingBox.bounding_boxes[i].ymin;
      yolo_num[i][4] = boundingBox.bounding_boxes[i].ymax;
      
    }

    mut_yolo.unlock();
  }
  else
  { }
}

void ImageLiDARFusion::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  // image correction
  if(IS_IMAGE_CORRECTION)
  {
    mut_img.lock();
    cv::undistort(cv_ptr->image, image_color, this->intrinsic_matrix, this->dist_matrix);
    // image_color = cv_ptr->image.clone(); 
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
  sensor_msgs::ImagePtr linshi_msg = cv_ptr->toImageMsg();
  linshi_msg->header.stamp = ros::Time::now();
  image_pub.publish(linshi_msg);

  frame1 = cv_bridge::toCvShare(msg, "bgr8")->image;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "fusion");
  if (argc != 2){ROS_ERROR("need is_correct as argument"); return -1;}; 
  
  if (strcmp(argv[1], "true") == 0){IS_IMAGE_CORRECTION = true; ROS_INFO("correct image");}
  else {IS_IMAGE_CORRECTION = false;ROS_INFO("don't correct image");}
  ImageLiDARFusion ic;
  ros::MultiThreadedSpinner spinner(4);  //노드 멀티쓰레딩
  spinner.spin();   
  // cv::waitKey();
  // ros::spin();
  return 0;
}