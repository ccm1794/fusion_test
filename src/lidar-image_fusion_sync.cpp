// 라이다-카메라 메세지를 message_filter를 통해 받아 이미지에 라이다 포인트 투영시키기
// 1. message_filter를 통해 같은 time_stamp 메세지를 받는다!
// 2. 라이다 포인트를 이미지에 투영시킨다!

#include <rclcpp/rclcpp.hpp>
#include <mutex>
#include <memory>
#include <thread>
#include <pthread.h>
#include <string>

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

// message_filter
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

using namespace std;

using std::placeholders::_1;
using std::placeholders::_2;

class ImageLiDARFusion : public rclcpp::Node
{
public:
  ImageLiDARFusion() 
  : Node("projection")
  {
    subscription_temp_1_.subscribe(this, "/video1");
    subscription_temp_2_.subscribe(this, "/velodyne_points");

    sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud2>>(subscription_temp_1_, subscription_temp_2_, 3);
    sync_->registerCallback(std::bind(&ImageLiDARFusion::topic_callback, this, _1, _2));
  }

public:
  void topic_callback(const sensor_msgs::msg::Image::ConstSharedPtr& tmp_1, const sensor_msgs::msg::PointCloud2::SharedPtr& tmp_2) const;

private:
  message_filters::Subscriber<sensor_msgs::msg::Image> subscription_temp_1_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> subscription_temp_2_;
  std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud2>> sync_;
};

void ImageLiDARFusion::topic_callback(const sensor_msgs::msg::Image::ConstSharedPtr& tmp_1, const sensor_msgs::msg::PointCloud2::SharedPtr& tmp_2) const
{
  cout << "오긴 한다?" << endl;
}


int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageLiDARFusion>());
  rclcpp::shutdown();

  return 0;
}