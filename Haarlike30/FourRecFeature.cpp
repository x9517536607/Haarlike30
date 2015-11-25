#include "FourRecFeature.h"


FourRecFeature::FourRecFeature(int scaleH, int scaleW, int x, int y)
{
    this->scaleH = scaleH;
    this->scaleW = scaleW;
    this->x = x;
    this->y = y;
    this->minW = 2;
    this->minH = 2;
}

Point FourRecFeature::getMINSize()
{
    return Point(2, 2);
}

FourRecFeature::~FourRecFeature()
{
}

float FourRecFeature::cal(Mat ConcatImage)
{
    /*
    ¢z¢w¢s¢w¢s¢w¢{
    ¢x1¢x2¢x3¢x
    ¢u¢w¢q¢w¢q¢w¢t
    ¢x4¢x5¢x6¢x+-
    ¢u¢w¢q¢w¢q¢w¢t
    ¢x7¢x8¢x9¢x-+
    ¢|¢w¢r¢w¢r¢w¢}
    */
    int curH = minH * scaleH;
    int curW = minW * scaleW;
    /* if (integralImage.cols < x + curW || integralImage.rows < y + curH || curH == 0 || curW == 0)
    return FLT_MIN;*/
    //Mat haar = Mat(curH, curW, CV_32F);
    //Mat ConcatImage = Tools::getConcatImage(integralImage);
    Mat srcROI(ConcatImage, Rect(x, y, curW + 1, curH + 1));//((_Tp*)(data + step.p[0]*i0))[i1]
    //top
    float p1 = srcROI.at<float>(0, 0);
    float p2 = srcROI.at<float>(0, (int)ceil((curW + 1) / 2.0) - 1);
    float p3 = srcROI.at<float>(0, curW);
    //mid
    float p4 = srcROI.at<float>((int)ceil((curH + 1) / 2.0) - 1, 0);
    float p5 = srcROI.at<float>((int)ceil((curH + 1) / 2.0) - 1, (int)ceil((curW + 1) / 2.0) - 1);
    float p6 = srcROI.at<float>((int)ceil((curH + 1) / 2.0) - 1, curW);
    //bot
    float p7 = srcROI.at<float>(curH, 0);
    float p8 = srcROI.at<float>(curH, (int)ceil((curW + 1) / 2.0) - 1);
    float p9 = srcROI.at<float>(curH, curW);
    //result = (p5 + p1 - p2 - p4) + (p9 + p5 - p6 - p8) - (p6 + p2 - p3 - p5) - (p8 + p4 - p5 - p7);
    result = (p9 + p1 - p3 - p7) - 2 * (p6 + p2 - p3 - p5) - 2 * (p8 + p4 - p5 - p7);
    return result;
}