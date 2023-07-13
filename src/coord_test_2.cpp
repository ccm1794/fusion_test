#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <vector>
#include <cmath>

#include <geometry_msgs/msg/point.hpp>
#include "std_msgs/msg/float32.hpp"
#include <visualization_msgs/msg/marker.hpp>


#define X first
#define Y second
#define PI  3.141592653589793238462643383279

sensor_msgs::msg::PointCloud2 colored_msg;

using PointT = pcl::PointXYZI;
using namespace std;


struct erpPoint { float x, y; };
float d_first_x = 0.0, d_first_y = -0.5; //왼쪽아래    //-1 -1  // 5km   // 0.5
float d_second_x = 0.0, d_second_y = -3.0; //오른쪽 아래 -1 -2           // 0.5

std::vector<pair<float, float>> right_ob_v, left_ob_v, coord_v, ob;

double yaw = 90 * (PI / 180);
double delta = 0.f;

class MyPoint
{
public:
  float X; float Y;
  MyPoint(float x, float y){
    X = x; Y = y;
  }
  void setCoord(float x, float y){
    X = x; Y = y;
  }
};


int ccw(const MyPoint &a, const MyPoint &b, const MyPoint &obs)
{
  float vec = (b.X - a.X) * (obs.Y - a.Y) - (b.Y - a.Y) * (obs.X - a.X);
  if (vec < 0)
    return -1;
  else if (vec > 0)
    return 1;
  else
    return 0;
}

class VehicleState
{
private:
	double cx; double cy;

public:
	VehicleState(double cx_, double cy_)	{
		cx = cx_; cy = cy_;
	}
	void Calculate()
	{
		double phi = 0.f; double deltaMax = 28 * (PI / 180);

		if (cy < 0)  phi = atan2(cx, -cy);

		else if (cy > 0) phi = PI - atan2(cx, cy);

		delta = phi - yaw;
    delta = delta*0.9;

		if (delta > deltaMax) delta = deltaMax;
		else if (delta < -deltaMax) delta = -deltaMax;
	}
};

float distance_zero(std::pair<float, float> p1){
    return sqrt(pow(p1.X,2) + pow(p1.Y,2));
} 

bool compare(std::pair<float, float> p1, std::pair<float, float> p2){
	if(distance_zero(p1) < distance_zero(p2)){
		// 길이가 짧은 경우 1순위로 정렬 
		return 1;
	}else if(distance_zero(p1) > distance_zero(p2)){
		return 0;
	}else{
		// 길이가 같은 경우  
		return 0;
	}
}

class SensorFusion : public rclcpp::Node
{
public:
  SensorFusion()
  : Node("SensorFusion")
  {
    pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("velodyne_cluster", 1);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("point", 10);
    marker_pub1_ = this->create_publisher<visualization_msgs::msg::Marker>("text", 10);
    delta_pub = this->create_publisher<std_msgs::msg::Float32>("delta", 10);
    tobe_colored_lidar_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("centerP_to_test01", 10);

    sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("velodyne_points", 1, std::bind(&SensorFusion::callback, this, std::placeholders::_1));
    XYZcolor_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("test_LiDAR", 10, std::bind(&SensorFusion::XYZcolor_callback, this, std::placeholders::_1));
  }

private:
  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr input) const
  {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>); 
  pcl::fromROSMsg(*input, *cloud); pcl::VoxelGrid<pcl::PointXYZI> VG; 
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>); 
  VG.setInputCloud(cloud);VG.setLeafSize(0.1f, 0.1f, 0.1f); VG.filter(*cloud_filtered); 
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered2 (new pcl::PointCloud<pcl::PointXYZI>); 
  pcl::CropBox<pcl::PointXYZI> cropFilter; cropFilter.setInputCloud(cloud_filtered);
  //===========================================[   R   O   I   영   역   ]=========================================================================
  // cropFilter.setMin(Eigen::Vector4f(-0, -4.5, -0.5, 0)); // x, y, z, min (m)    
  // cropFilter.setMax(Eigen::Vector4f(9, 4.5, 1.0, 0));        // (-0,20) (-10,10) (-0.5,1.0)...............8/4 data

  cropFilter.setMin(Eigen::Vector4f(-0, -8, -0.5, 0)); // x, y, z, min (m)    
  cropFilter.setMax(Eigen::Vector4f(9, 8, 1.0, 0));

  cropFilter.filter(*cloud_filtered2);pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_erp (new pcl::PointCloud<pcl::PointXYZI>); 
  vector<erpPoint> erpVec; 
  pcl::PointIndices::Ptr erpIndices(new pcl::PointIndices);

  for(auto iter = cloud_filtered2->begin();iter!=cloud_filtered2->end();++iter)
  {
    if(iter->z>0.5 )
    {
      float x = iter->x; float y = iter->y; 
      erpPoint erp; erp.x = x; erp.y = y; 
      bool erppoint = false;
      
      for(auto it = erpVec.begin(); it!= erpVec.end(); ++it)
      {
        if(abs(it->y - y) < 2 && abs(it->x - x)< 2){ erppoint = true;  break; }
      }
      if(!erppoint)   erpVec.push_back(erp);
    }
  }
  int i=0; pcl::ExtractIndices<pcl::PointXYZI> ext;
  for(auto iter = cloud_filtered2->begin();iter!=cloud_filtered2->end();++iter)
  {
    for(auto it = erpVec.begin(); it!= erpVec.end();++it)
    {
      float x = it->x;  float y = it->y;
      
      if(abs(iter->y - y) < 2 && abs(iter->x - x)< 2)
      {
        iter->intensity = 100;  erpIndices->indices.push_back(i);  break;
      }
    }
    ++i;
  }

  ext.setInputCloud(cloud_filtered2); ext.setIndices(erpIndices); ext.setNegative(true); ext.filter(*cloud_erp); 
  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new 	pcl::search::KdTree<pcl::PointXYZI>); 
  tree->setInputCloud (cloud_erp);  vector<pcl::PointIndices> cluster_indices; 
  pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec; ec.setInputCloud (cloud_erp); 
  ec.setClusterTolerance (0.6); ec.setMinClusterSize (1); ec.setMaxClusterSize (1000); ec.setSearchMethod (tree); ec.extract (cluster_indices);

  int ii = 0; pcl::PointCloud<PointT> TotalCloud; std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr > > clusters;
  for(vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it, ++ii)
  {
    pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>);
    for(vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
    { 
      cluster->points.push_back(cloud_erp->points[*pit]); 
      PointT pt = cloud_erp->points[*pit]; PointT pt2; 
      pt2.x = pt.x, pt2.y = pt.y, pt2.z = pt.z; pt2.intensity = (float)(ii + 1);TotalCloud.push_back(pt2);
    }
    cluster->width = cluster->size(); cluster->height = 1; cluster->is_dense = true; clusters.push_back(cluster);
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc_gotohell(new pcl::PointCloud<pcl::PointXYZ>);
  for(int i =0; i<clusters.size(); i++)
  {

    Eigen::Vector4f centroid, min_p, max_p;
    pcl::compute3DCentroid(*clusters[i], centroid);
    pcl::getMinMax3D(*clusters[i], min_p, max_p);
    geometry_msgs::msg::Point center_point;
    center_point.x = centroid[0]; center_point.y = centroid[1]; center_point.z = centroid[2];
    geometry_msgs::msg::Point min_point;
    min_point.x = min_p[0]; min_point.y = min_p[1]; min_point.z = min_p[2];
    geometry_msgs::msg::Point max_point;
    max_point.x = max_p[0]; max_point.y = max_p[1]; max_point.z = max_p[2];

    float ob_size = sqrt(pow(max_point.x - min_point.x, 2) + pow(max_point.y - min_point.y, 2));

    if(center_point.x>=0.5 && center_point.x<=10 && center_point.y<=7 && center_point.y>=-7 && max_point.z<0.2 && max_point.z>-0.4 && ob_size<0.3 && ob_size >0.05)
    {

      pcl::PointXYZ gotohell;
      gotohell.x = center_point.x;
      gotohell.y = center_point.y;
      gotohell.z = center_point.z;
      pc_gotohell->push_back(gotohell);   
    }
    pc_gotohell->width = 1;
    pc_gotohell->height = pc_gotohell->points.size();
    pcl::toROSMsg(*pc_gotohell, colored_msg); 
    colored_msg.header.frame_id = "velodyne";
    //colored_msg.header.stamp = rclcpp::Clock::now(); 
    tobe_colored_lidar_pub->publish(colored_msg);
  }
  pcl::PCLPointCloud2 cloud_p;
  pcl::toPCLPointCloud2(TotalCloud, cloud_p);
  sensor_msgs::msg::PointCloud2 output; 
  pcl_conversions::fromPCL(cloud_p, output);
  output.header.frame_id = "velodyne";
  pub->publish(output);
  }

  void XYZcolor_callback(const sensor_msgs::msg::PointCloud2::SharedPtr XYZcolor) const
  {
    std::pair<float, float> right_ob,left_ob, coord, object; right_ob_v.clear(); left_ob_v.clear();
  pcl::PointCloud<pcl::PointXYZI>::Ptr input (new pcl::PointCloud<pcl::PointXYZI>); pcl::fromROSMsg(*XYZcolor, *input);
  for(int i = 0 ; i<input->points.size(); i++ )
  {
    PointT input_cloud; input_cloud.x = input->points[i].x; input_cloud.y = input->points[i].y; input_cloud.intensity = input->points[i].intensity;
    //geometry_msgs::msg::Point cloud; cloud.x = input_cloud.x; cloud.y = input_cloud.y; cloud.intensity = input_cloud.intensity;
    // std::cout << "input : "<< input_cloud.x <<  "," << input_cloud.y << "color : " << input_cloud.intensity << endl;
    object = make_pair(input_cloud.x, input_cloud.y); ob.push_back(object);
    
    // if(input_cloud.y >0)//라이다 기준 라바콘 점이 양수
    // {
      if(input_cloud.intensity == 0 || !input_cloud.intensity )//none
      {
        if(input_cloud.y >0)
        {
          left_ob = make_pair(input_cloud.x, input_cloud.y); left_ob_v.push_back(left_ob);
        }
        else if(input_cloud.y <=0)
        {
          right_ob = make_pair(input_cloud.x, input_cloud.y); right_ob_v.push_back(right_ob); 
        }
        // for(int i = 0 ; i < left_ob_v.size() ; i++)
        // {
        //   std::cout << i+1 << "번째 : " << left_ob_v[i].X << "  ,  " << left_ob_v[i].Y << '\n';
        // }
      }
      else if(input_cloud.intensity == 1)//파랑
      {
        right_ob = make_pair(input_cloud.x, input_cloud.y); right_ob_v.push_back(right_ob);
      }
      else if(input_cloud.intensity == 2)//노랑
      {
        left_ob = make_pair(input_cloud.x, input_cloud.y); left_ob_v.push_back(left_ob);
      }
    // else if(input_cloud.y < 0)//음수
    // {
    //   if(input_cloud.intensity == 0)
    //   {
    //     right_ob = make_pair(input_cloud.x, input_cloud.y); right_ob_v.push_back(left_ob);
    //   }
    //   else if(input_cloud.intensity == 1)
    //   {
    //     right_ob = make_pair(input_cloud.x, input_cloud.y); right_ob_v.push_back(left_ob);
    //   }
    //   else if(input_cloud.intensity == 2)
    //   {
    //     left_ob = make_pair(input_cloud.x, input_cloud.y); left_ob_v.push_back(left_ob);
    //   }
    sort(right_ob_v.begin(),right_ob_v.end(),compare);  sort(left_ob_v.begin(),left_ob_v.end(),compare);
  }
    //cout << "color : " <<input_cloud.intensity << endl;
    
  
  coord_v.clear();
  if(right_ob_v.size() == 0 && left_ob_v.size() == 0){
    return;
  }
  else if(right_ob_v.size() != 0 && left_ob_v.size() != 0){        
    if(right_ob_v.size() > left_ob_v.size()){
      for(int i = 0; i < left_ob_v.size(); i++){
        for(int j = 0; j < right_ob_v.size(); j++){
          float coord_x = 0;                     
          if(i == j){
            coord_x = (left_ob_v[i].X + right_ob_v[j].X )/2.0;       coord = make_pair(coord_x,(left_ob_v[i].Y + right_ob_v[j].Y )/2.0);     coord_v.push_back(coord);
          }
          if(i == left_ob_v.size()-1 && left_ob_v.size() - 1 < j){
            coord_x = (left_ob_v[i].X + right_ob_v[j].X )/2.0;        coord = make_pair(coord_x,(left_ob_v[i].Y + right_ob_v[j].Y )/2.0);     coord_v.push_back(coord);                      
          }
        }
      }    
    }
    else{
      for(int i = 0; i < right_ob_v.size(); i++){
        for(int j = 0; j < left_ob_v.size(); j++){
          float coord_x = 0;
          if(i == j){
            coord_x = (right_ob_v[i].X + left_ob_v[j].X )/2.0;    coord = make_pair(coord_x,(right_ob_v[i].Y + left_ob_v[j].Y )/2.0);    coord_v.push_back(coord);
          }
                          
          if(i == right_ob_v.size()-1 && right_ob_v.size() - 1 < j){
            coord_x = (right_ob_v[i].X + left_ob_v[j].X )/2.0;     coord = make_pair(coord_x,(right_ob_v[i].Y + left_ob_v[j].Y )/2.0);   coord_v.push_back(coord);                       
          }              
        }
      }
    }
  }
  else if(right_ob_v.size() != 0 && left_ob_v.size() == 0){
    for(int i = 0 ; i < right_ob_v.size(); i++){ coord = make_pair(right_ob_v[i].X, right_ob_v[i].Y + 1.8);    coord_v.push_back(coord); }
  }
  else if(right_ob_v.size() == 0 && left_ob_v.size() != 0){
    for(int i = 0 ; i < left_ob_v.size(); i++){ coord = make_pair(left_ob_v[i].X, left_ob_v[i].Y - 1.8);         coord_v.push_back(coord);}
  }
    
  for(int i = 0; i < input->points.size(); i++)
  {
    std::cout << "cluster__Push[" <<i+1<< "] : " << ob[i].X << " , " << ob[i].Y << '\n';
  } 
  std::cout << "===========[왼쪽 벡터]" << '\n';
  
  for(int i = 0; i < left_ob_v.size(); i++)
  {
    std::cout << i+1 << "번째 : " << left_ob_v[i].X << "  ,  " << left_ob_v[i].Y << '\n';
  } 
  std::cout << "===========[오른쪽 벡터]" << '\n'; 
  for(int i = 0; i < right_ob_v.size(); i++)
  { 
    std::cout << i+1 << "번째 : " << right_ob_v[i].X << "  ,  "<<right_ob_v[i].Y << '\n';
  }
  
  geometry_msgs::msg::Point coord_point;     coord_point.x = coord_v.front().X;        coord_point.y = coord_v.front().Y;          coord_point.z = 0.0;
  VehicleState state(coord_point.x, coord_point.y); state.Calculate();          delta = delta * (180 / PI)*71;          
  std_msgs::msg::Float32 delta_; delta_.data = delta; delta_pub->publish(delta_);
  

  cout<<"=======================각도: "<< (atan2(coord_point.y,coord_point.x))<<'\n';
  std::cout << "===========[경로]===========" << '\n';       
  std::cout  << "cx:" << coord_point.x << " , " << "cy:"<<coord_point.y <<'\n'; 
  std::cout << "===========[delta]===========" << '\n';      
  std::cout  <<"delta: "<<delta<<'\n';

  geometry_msgs::msg::Point p;
  visualization_msgs::msg::Marker marker1; 
  marker1.ns  ="points_and_lines";
  marker1.action = visualization_msgs::msg::Marker::ADD;
  marker1.type = visualization_msgs::msg::Marker::POINTS;
  marker1.id = 0;
  marker1.pose.orientation.w = 1.0;
  marker1.scale.x = 0.1;
  marker1.scale.y = 0.1;
  marker1.color.a = 1.0;
  marker1.color.b = 1.0f;
  p.x = coord_point.x;p.y = coord_point.y;
  p.z = 0.0; marker1.points.push_back(p);
  visualization_msgs::msg::Marker node_name;
  node_name.text = std::to_string(delta); 
  node_name.color.a = 1.0; 
  node_name.color.b = 1.0f;
  node_name.scale.z = 0.5; 
  node_name.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING; 
  node_name.id = 0; 
  node_name.action = visualization_msgs::msg::Marker::ADD; 
  node_name.pose.orientation.w = 1.0; 
  node_name.pose.position.x = p.x;  
  node_name.pose.position.y = p.y;
  marker1.header.frame_id = node_name.header.frame_id = "velodyne";
  //marker1.header.stamp  = node_name.header.stamp = rclcpp::Clock::now();	
  marker_pub_->publish(marker1);
  marker_pub1_->publish(node_name);
  
  }
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub1_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr delta_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tobe_colored_lidar_pub;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr XYZcolor_sub;
};

int main (int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorFusion>());
  rclcpp::shutdown();
  return 0;
}