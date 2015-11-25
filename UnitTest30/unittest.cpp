#pragma once
#include <SDKDDKVer.h>
#include "CppUnitTest.h"
#include "../Haarlike30/Tools.h"
#include "../Haarlike30/EdgeFeature.h"
#include "../Haarlike30/EdgeFeature2.h"
#include "../Haarlike30/FourRecFeature.h"
#include "../Haarlike30/LineFeature.h"
#include "../Haarlike30/LineFeature2.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTest
{
TEST_CLASS(UnitTest)
{
public:
    TEST_METHOD(IIG)
    {
        Mat src(Size(10, 10), CV_32F);
        src = Scalar::all(1);
        Mat dist = Tools::convertToIntegralImage(src);
        Assert::AreEqual(dist.at<float>(4, 8), (float)32);
    }
    TEST_METHOD(FourRecFeatureAllOneTest)
    {
        Mat src(Size(10, 10), CV_32F);
        src = Scalar::all(1);
        Mat dist = Tools::convertToIntegralImage(src);
        int featureMinW = FourRecFeature::getMINSize().x;
        int featureMinH = FourRecFeature::getMINSize().y;
        for (int h = 1; h <= 10 / featureMinH; h++)
        {
            for (int w = 1; w <= 10 / featureMinW; w++)
            {
                for (int y = 0; y <= 10 - featureMinH * h; y++)
                {
                    for (int x = 0; x <= 10 - featureMinW * w; x++)
                    {
                        FourRecFeature fourRecFeature(h, w, x, y);
                        Assert::AreEqual((float)0, fourRecFeature.cal(dist));
                    }
                }
            }
        }
    }
    TEST_METHOD(EdgeFeatureAllOneTest)
    {
        Mat src(Size(10, 10), CV_32F);
        src = Scalar::all(1);
        Mat dist = Tools::convertToIntegralImage(src);
        int featureMinW = EdgeFeature::getMINSize().x;
        int featureMinH = EdgeFeature::getMINSize().y;
        for (int h = 1; h <= 10 / featureMinH; h++)
        {
            for (int w = 1; w <= 10 / featureMinW; w++)
            {
                for (int y = 0; y <= 10 - featureMinH * h; y++)
                {
                    for (int x = 0; x <= 10 - featureMinW * w; x++)
                    {
                        EdgeFeature edgeFeature(h, w, x, y);
                        Assert::AreEqual((float)0, edgeFeature.cal(dist));
                    }
                }
            }
        }
    }
    TEST_METHOD(EdgeFeature2AllOneTest)
    {
        Mat src(Size(10, 10), CV_32F);
        src = Scalar::all(1);
        Mat dist = Tools::convertToIntegralImage(src);
        int featureMinW = EdgeFeature2::getMINSize().x;
        int featureMinH = EdgeFeature2::getMINSize().y;
        for (int h = 1; h <= 10 / featureMinH; h++)
        {
            for (int w = 1; w <= 10 / featureMinW; w++)
            {
                for (int y = 0; y <= 10 - featureMinH * h; y++)
                {
                    for (int x = 0; x <= 10 - featureMinW * w; x++)
                    {
                        EdgeFeature2 edgeFeature2(h, w, x, y);
                        Assert::AreEqual((float)0, edgeFeature2.cal(dist));
                    }
                }
            }
        }
    }
    TEST_METHOD(LineFeatureAllOneTest)
    {
        Mat src(Size(10, 10), CV_32F);
        src = Scalar::all(1);
        Mat dist = Tools::convertToIntegralImage(src);
        int featureMinW = LineFeature::getMINSize().x;
        int featureMinH = LineFeature::getMINSize().y;
        for (int h = 1; h <= 10 / featureMinH; h++)
        {
            for (int w = 1; w <= 10 / featureMinW; w++)
            {
                for (int y = 0; y <= 10 - featureMinH * h; y++)
                {
                    for (int x = 0; x <= 10 - featureMinW * w; x++)
                    {
                        LineFeature lineFeature(h, w, x, y);
                        Assert::AreEqual((float)  w * h, lineFeature.cal(dist));
                    }
                }
            }
        }
    }
    TEST_METHOD(LineFeature2AllOneTest)
    {
        Mat src(Size(10, 10), CV_32F);
        src = Scalar::all(1);
        Mat dist = Tools::convertToIntegralImage(src);
        int featureMinW = LineFeature2::getMINSize().x;
        int featureMinH = LineFeature2::getMINSize().y;
        for (int h = 1; h <= 10 / featureMinH; h++)
        {
            for (int w = 1; w <= 10 / featureMinW; w++)
            {
                for (int y = 0; y <= 10 - featureMinH * h; y++)
                {
                    for (int x = 0; x <= 10 - featureMinW * w; x++)
                    {
                        LineFeature2 lineFeature2(h, w, x, y);
                        Assert::AreEqual((float)w * h, lineFeature2.cal(dist));
                    }
                }
            }
        }
    }

	TEST_METHOD(getResultMatrix) 
	{
		float arr1[] = { 1,6,10,8,4,3,9 };
		vector<float> d1(arr1, arr1 + sizeof(arr1) / sizeof(arr1[0]));
		float arr2[] = { 9,2,7,10,4,11,12 };
		vector<float> d2(arr2, arr2 + sizeof(arr2) / sizeof(arr2[0]));
		HaarValue test(1,Size(1,1),Point(0,0));
		test.NEGFVs = d2;
		test.POSFVs = d1;
		vector<vector<int>> RM;
		MyWeakClassifier *wc1 = new MyWeakClassifier(1, 5);
		MyWeakClassifier *wc2 = new MyWeakClassifier(1, 8);
		Tools::getResultMatrix(wc1, test, RM);
		Tools::getResultMatrix(wc2, test, RM);
		Assert::AreEqual(1, RM[0][0]);
		Assert::AreEqual(-1, RM[0][2]);
		Assert::AreEqual(1, RM[0][4]);
		Assert::AreEqual(-1, RM[0][6]);
		//
		Assert::AreEqual(-1, RM[1][7]);
		Assert::AreEqual(1, RM[1][8]);
		Assert::AreEqual(1, RM[1][9]);
		Assert::AreEqual(1, RM[1][11]);
	}

    TEST_METHOD(feature)
    {
        float data[100];
        for (int i = 0; i < 100; i++)
        {
            data[i] = i + 1;
        }
        Mat src(Size(10, 10), CV_32F, &data);
        Assert::AreEqual(src.at<float>(2, 7), data[2 * 10 + 7]);
        Mat dist = Tools::convertToIntegralImage(src);
        EdgeFeature f1(2, 2, 2, 2);
        Assert::AreEqual(f1.cal(dist), (float) - 8);
        EdgeFeature f2(1, 1, 3, 4);
        Assert::AreEqual(f2.cal(dist), (float) - 1);
        LineFeature2 f3(1, 1, 0, 0);
        Assert::AreEqual(f3.cal(dist), (float)2);
        LineFeature2 f4(1, 2, 0, 0);
        Assert::AreEqual(f4.cal(dist), (float)7);
        LineFeature2 f5(1, 2, 1, 0);
        Assert::AreEqual(f5.cal(dist), (float)9);
        LineFeature f6(1, 1, 0, 0);
        Assert::AreEqual(f6.cal(dist), (float)11);
        LineFeature f7(1, 2, 0, 0);
        Assert::AreEqual(f7.cal(dist), (float)23);
        LineFeature f8(1, 2, 1, 0);
        Assert::AreEqual(f8.cal(dist), (float)25);
        FourRecFeature f9(1, 1, 0, 0);
        Assert::AreEqual(f9.cal(dist), (float)0);
        float data2[] = {1, 1, 0, 0, 0, 0, 1, 1};
        Mat src2(Size(4, 2), CV_32F, &data2);
        Mat dist2 = Tools::convertToIntegralImage(src2);
        FourRecFeature f10(1, 2, 0, 0);
        Assert::AreEqual(f10.cal(dist2), (float)4);
    }
};
}