#pragma once
#ifndef TOOLS
#define TOOLS 1

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "EdgeFeature.h"
#include "EdgeFeature2.h"
#include "LineFeature.h"
#include "FourRecFeature.h"
#include "LineFeature2.h"
#include <boost/filesystem.hpp>
#include "adaBoost/adaboost.hpp"
#include "adaBoost/weak_classifier.hpp"

using namespace cv;
using namespace std;
using namespace DM_AG;

class HaarValue
{
    public:
        HaarValue(int t, Size size, Point point)
            : type(t), s(size), p(point) {}
        ~HaarValue() {}
        vector<float> POSFVs;
        vector<float> NEGFVs;
        int type;
        Size s;
        Point p;
        float posThreshold;
        float negThreshold;
        int pSign;
};

namespace Tools 
{
	inline Mat getConcatImage(Mat src)
	{
		Mat res(src.rows + 1, src.cols + 1, CV_32FC1, Scalar(0));
		Mat ROI(res, Rect(1, 1, src.cols, src.rows));
		src.copyTo(ROI);
		return res;
	}
	
	inline Mat convertToIntegralImage(Mat src)
	{
		if (src.channels() == 3)
			cvtColor(src, src, CV_RGB2GRAY);
		Mat concatImage = getConcatImage(src);
		Mat integralImage = concatImage.clone();
		for (int y = 1; y < integralImage.rows; y++)
		{
			for (int x = 1; x < integralImage.cols; x++)
			{
				integralImage.at<float>(y, x)
					= integralImage.at<float>(y, x) 
					+ integralImage.at<float>(y - 1, x)
					+ integralImage.at<float>(y, x - 1)
					- integralImage.at<float>(y - 1, x - 1);
			}
		}
		//Mat ROI(integralImage, Rect(1, 1, src.cols, src.rows));
		return integralImage.clone();
	}

	

	inline void printImage(Mat src)
	{
		for (int y = 0; y < src.rows; y++)
		{
			for (int x = 0; x < src.cols; x++)
			{
				if (src.type() == CV_32F)
					cout << src.at<float>(y, x) << " ";
				else if (src.type() == CV_8U)
					cout << (int)src.at<uchar>(y, x) << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	inline vector<Mat> LoadDirectory(string directory)
	{
		vector<Mat> allImageInDirectory;
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
					allImageInDirectory.push_back(src);
				}
			}
		}
		return allImageInDirectory;
	}

	inline vector<Mat> getIIGImages(vector<Mat> Mats)
	{
		vector<Mat> IIGMats;
		for (int i = 0; i < Mats.size(); i++)
		{
			IIGMats.push_back(convertToIntegralImage(Mats[i]));
		}
		return IIGMats;
	}

	inline float getMean(vector<float> FVs)
	{
		int FVSize = FVs.size();
		float sum = 0;
		for (int i = 0; i < FVSize; i++)
		{
			sum += FVs[i];
		}
		return sum / FVSize;
	}

	inline void getResultMatrix(Classifier<float> *wc, HaarValue &haarValue, vector<vector<int>>& rm)
	{
		int posSize = haarValue.POSFVs.size();
		int negSize = haarValue.NEGFVs.size();
		vector<int> result;
		for (int i = 0; i < posSize; i++)
		{
			result.push_back((*wc).analyze(haarValue.POSFVs[i]));
		}
		for (int i = 0; i < negSize; i++)
		{
			result.push_back((*wc).analyze(haarValue.NEGFVs[i]));
		}
		rm.push_back(result);
	}

	template<class T>
	inline void getAllHaarFeature(vector<HaarValue>& allfeature, const vector<Mat>& posIIG, const vector<Mat>& negIIG)
	{
		int simpleH = posIIG[0].rows - 1;
		int simpleW = posIIG[0].cols - 1;
		int minH = T::getMINSize().y;
		int minW = T::getMINSize().x;
		int maxScaleH = simpleH / minH;
		int maxScaleW = simpleW / minW;
		int posCount = posIIG.size();
		int negCount = negIIG.size();
		for (int h = 1; h <= maxScaleH; h++)
		{
			for (int w = 1; w <= maxScaleW; w++)
			{
				for (int y = 0; y <= simpleH - minH * h; y++)
				{
					for (int x = 0; x <= simpleW - minW * w; x++)
					{
						T e = T(h, w, x, y);
						HaarValue haarValue(e.getType(), Size(w, h), Point(x, y));
						for (int i = 0; i < posCount; i++)
						{
							haarValue.POSFVs.push_back(e.cal(posIIG[i]));
						}
						for (int i = 0; i < negCount; i++)
						{
							haarValue.NEGFVs.push_back(e.cal(negIIG[i]));
						}
						allfeature.push_back(haarValue);
					}
				}
			}
		}
	}

	inline string toLowerCase(const string& in)
	{
		string t;
		for (string::const_iterator i = in.begin(); i != in.end(); ++i)
		{
			t += tolower(*i);
		}
		return t;
	}

	inline float getVectorMean(vector<float>& h)
	{
		int featureSize = h.size();
		float sum = 0;
		for (int i = 0; i < featureSize; i++)
		{
			sum += h[i];
		}
		return sum / (float)featureSize;
	}

	inline void stringSplit(string str, string separator, vector<string>* results)//¦r¦ê¤Á³Î
	{
		results->clear();
		int found;
		found = str.find_first_of(separator);
		while (found != string::npos)
		{
			if (found > 0)
			{
				results->push_back(str.substr(0, found));
			}
			str = str.substr(found + 1);
			found = str.find_first_of(separator);
		}
		if (str.length() > 0)
		{
			results->push_back(str);
		}
	}
}

#endif