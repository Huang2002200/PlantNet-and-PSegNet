#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/math/special_functions/round.hpp>
#include <pcl/surface/mls.h>        //最小二乘法平滑处理类定义头文件

#include <pcl/io/pcd_io.h>
#include <pcl/surface/mls.h>        //最小二乘法平滑处理类定义头文件
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/grid_projection.h>
#include <iostream>
#include <string.h>
#include <vector>
#include <pcl/visualization/pcl_visualizer.h>
#include <time.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/features/boundary.h>

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
using namespace std;

void getFiles(string path, vector<string>& files)
{
	
	intptr_t hFile = 0;

	struct _finddata_t fileinfo;

	string search_path = path + "/*.txt";
	if ((hFile = _findfirst(search_path.c_str(), &fileinfo)) != -1)
	{
		do
		{
			string filename = path + "/" + fileinfo.name;
			files.push_back(filename);
		} while (_findnext(hFile, &fileinfo) == 0); 

		_findclose(hFile);
	}
	else
	{
		cout << "No .txt files found in directory: " << path << endl;
	}
}

void readTxtToPointCloud(const string& filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	ifstream infile(filename);
	if (infile)
	{
		pcl::PointXYZRGB point;
		string line;
		while (getline(infile, line))
		{
			istringstream iss(line);
			float r, g, b;
			// Read x, y, z, r, g, b
			if (!(iss >> point.x >> point.y >> point.z >> r >> g >> b))
				break;

			point.r = static_cast<uint8_t>(r);
			point.g = static_cast<uint8_t>(g);
			point.b = static_cast<uint8_t>(b);

			cloud->push_back(point);
		}
		infile.close();
	}
	else
	{
		cerr << "Error opening file: " << filename << endl;
	}
}

bool savePointCloudToTxt(const string& filename, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
	ofstream outfile(filename);
	if (!outfile.is_open()) {
		cerr << "Error opening file: " << filename << endl;
		return false;
	}

	for (size_t i = 0; i < cloud->size(); ++i) {
		const pcl::PointXYZRGB& point = cloud->points[i];
		outfile << point.x << " " << point.y << " " << point.z << " "
			<< static_cast<int>(point.r) << " " << static_cast<int>(point.g) << " " << static_cast<int>(point.b) << endl;
	}

	outfile.close();
	cout << "Saved point cloud to: " << filename << endl;
	return true;
}

int main(int argc, char** argv)
{
	vector<string> files;
	string filePath = "/raw_data_path";
	// 获取该路径下的所有文件
	getFiles(filePath, files);

	for (const auto& file : files)
	{
		ifstream infile(file);
		if (infile)
		{
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudOrigin(new pcl::PointCloud<pcl::PointXYZRGB>);
			readTxtToPointCloud(file, cloudOrigin);

			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_b(new pcl::PointCloud<pcl::PointXYZRGB>);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_c(new pcl::PointCloud<pcl::PointXYZRGB>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
			//  cloudOrigin  xyz information copy to cloud
			for (size_t i = 0; i < cloudOrigin->size(); ++i)
			{
				pcl::PointXYZ point;
				point.x = cloudOrigin->points[i].x;
				point.y = cloudOrigin->points[i].y;
				point.z = cloudOrigin->points[i].z;
				cloud->push_back(point);
			}
			//计算法线
			pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;
			pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
			pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
			pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);

			tree->setInputCloud(cloud);
			normEst.setInputCloud(cloud);
			normEst.setSearchMethod(tree);
			normEst.setKSearch(20);
			normEst.compute(*normals);
			//判断边缘点
			pcl::PointCloud<pcl::Boundary> boundaries;
			pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundEst;
			tree2->setInputCloud(cloud);
			boundEst.setInputCloud(cloud);
			boundEst.setInputNormals(normals);
			boundEst.setSearchMethod(tree2);
			boundEst.setKSearch(20);
			boundEst.setAngleThreshold(M_PI / 2);
			boundEst.compute(boundaries);
			//提取边缘点重组点云
			cloud_b->width = cloudOrigin->points.size();
			cloud_b->height = 1;
			cloud_b->points.resize(cloud_b->width * cloud_b->height);
			//提取非边缘点重组点云
			cloud_c->width = cloudOrigin->points.size();
			cloud_c->height = 1;
			cloud_c->points.resize(cloud_c->width * cloud_c->height);
			int j = 0;
			int k = 0;
			for (int i = 0; i < cloudOrigin->points.size(); i++)
			{
				if (boundaries.points[i].boundary_point != 0)
				{
					cloud_b->points[j].x = cloudOrigin->points[i].x;
					cloud_b->points[j].y = cloudOrigin->points[i].y;
					cloud_b->points[j].z = cloudOrigin->points[i].z;
					cloud_b->points[j].r = cloudOrigin->points[i].r;
					cloud_b->points[j].g = cloudOrigin->points[i].g;
					cloud_b->points[j].b = cloudOrigin->points[i].b;
					j++;
				}
				else
				{
					cloud_c->points[k].x = cloudOrigin->points[i].x;
					cloud_c->points[k].y = cloudOrigin->points[i].y;
					cloud_c->points[k].z = cloudOrigin->points[i].z;
					cloud_c->points[k].r = cloudOrigin->points[i].r;
					cloud_c->points[k].g = cloudOrigin->points[i].g;
					cloud_c->points[k].b = cloudOrigin->points[i].b;
					k++;
				}
				continue;
			}
			cloud_b->width = j;
			cloud_b->points.resize(cloud_b->width * cloud_b->height);
			cloud_c->width = k;
			cloud_c->points.resize(cloud_c->width * cloud_c->height);
			cout << "raw point number" << cloudOrigin->size() << endl;
			cout << "edge point number" << cloud_b->size() << endl;

			//save files
			string path_e = "/edge_points_save_path";
			string path_c = "/core_points_save_path";
			string filename = file.substr(file.find_last_of("/\\") + 1);
			string filename_noext = filename.substr(0, filename.find_last_of("."));
			path_e += filename_noext + "_e.txt";
			path_c += filename_noext + "_c.txt";
			if (!savePointCloudToTxt(path_e, cloud_b) || !savePointCloudToTxt(path_c, cloud_c)) {
				cerr << "Error: Failed to save point cloud to txt files." << endl;
				return -1;
			}
		}
		infile.close();
	}
	return (0);
}