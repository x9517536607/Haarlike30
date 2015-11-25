#include "DataManager.h"



DataManager::DataManager(string Dpath)
{
	directory = Dpath;
	LoadDirectory();
	getIIGImages();
	oriMats.clear();
}


DataManager::~DataManager()
{
	oriMats.clear();
	iigMats.clear();
}

void DataManager::getIIGImages() 
{
	int oriSize = oriMats.size();
	for (int i = 0; i < oriSize; i++)
	{
		iigMats.push_back(Tools::convertToIntegralImage(oriMats[i]));
	}
}

void DataManager::LoadDirectory() 
{
	namespace fs = boost::filesystem;
	fs::path someDir(directory);
	fs::directory_iterator end_iter;
	if (fs::exists(someDir) && fs::is_directory(someDir))
	{
		for (fs::directory_iterator dir_iter(someDir); dir_iter != end_iter; ++dir_iter)
		{
			if (fs::is_regular_file(dir_iter->status()))
			{
				Mat src = imread(dir_iter->path().string(), 0);
				resize(src, src, cv::Size(24, 24));
				float k[9] = { 0, -1.0, 0, -1.0, 5.0, -1.0, 0, -1.0, 0 };
				CvMat km = cvMat(3, 3, CV_32FC1, k);
				cvFilter2D(&((IplImage)src), &((IplImage)src), &km, cvPoint(-1, -1));
				src.convertTo(src, CV_32F);
				oriMats.push_back(src);
			}
		}
	}
}
