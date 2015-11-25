#include "EdgeFeature.h"


EdgeFeature::EdgeFeature(int scaleH, int scaleW, int x, int y)
{
    this->scaleH = scaleH;
    this->scaleW = scaleW;
    this->x = x;
    this->y = y;
    this->minW = 2;
    this->minH = 1;
}

Point EdgeFeature::getMINSize()
{
    return Point(2, 1);
}

EdgeFeature::~EdgeFeature()
{
}

float EdgeFeature::cal(Mat ConcatImage)
{
    /*
    ¢z¢w¢s¢w¢s¢w¢{
    ¢x1¢x2¢x3¢x
    ¢u¢w¢q¢w¢q¢w¢t
    ¢x6¢x5¢x4¢x
    ¢|¢w¢r¢w¢r¢w¢}
    */
    int curH = minH * scaleH;
    int curW = minW * scaleW;
    /* if (integralImage.cols < x + curW || integralImage.rows < y + curH || curH == 0 || curW == 0)
         return FLT_MIN;*/
    //Mat haar = Mat(curH, curW, CV_32F);
    //Mat ConcatImage = Tools::getConcatImage(integralImage);
    Mat srcROI(ConcatImage, Rect(x, y, curW + 1, curH + 1));//((_Tp*)(data + step.p[0]*i0))[i1]
    float p1 = srcROI.at<float>(0, 0);
    float p2 = srcROI.at<float>(0, (int)ceil((curW + 1) / 2.0) - 1);
    float p3 = srcROI.at<float>(0, curW);
    float p4 = srcROI.at<float>(curH, curW);
    float p5 = srcROI.at<float>(curH, (int)ceil((curW + 1) / 2.0) - 1);
    float p6 = srcROI.at<float>(curH, 0);
    result = (p5 - p2 - p6 + p1) - (p4 - p5 - p3 + p2);
    return result;
}
