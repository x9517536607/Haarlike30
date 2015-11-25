#include "LineFeature.h"


LineFeature::LineFeature(int scaleH, int scaleW, int x, int y)
{
    this->scaleH = scaleH;
    this->scaleW = scaleW;
    this->x = x;
    this->y = y;
    this->minW = 1;
    this->minH = 3;
}

Point LineFeature::getMINSize()
{
    return Point(1, 3);
}

float LineFeature::cal(Mat ConcatImage)
{
    /*
    ¢z¢w¢s¢w¢{
    ¢x1¢x2¢x
    ¢u¢w¢q¢w¢t
    ¢x3¢x4¢x+
    ¢u¢w¢q¢w¢t
    ¢x5¢x6¢x-
    ¢u¢w¢q¢w¢t
    ¢x7¢x8¢x+
    ¢|¢w¢r¢w¢}
    */
    int curH = minH * scaleH;
    int curW = minW * scaleW;
    /* if (integralImage.cols < x + curW || integralImage.rows < y + curH || curH == 0 || curW == 0)
    return FLT_MIN;*/
    //Mat haar = Mat(curH, curW, CV_32F);
    //Mat ConcatImage = Tools::getConcatImage(integralImage);
    Mat srcROI(ConcatImage, Rect(x, y, curW + 1, curH + 1));
    //top
    float p1 = srcROI.at<float>(0, 0);
    float p2 = srcROI.at<float>(0, curW);
    //t_mid
    float p3 = srcROI.at<float>((int)ceil((curH + 1) / 3.0) - 1 , 0);
    float p4 = srcROI.at<float>((int)ceil((curH + 1) / 3.0) - 1 , curW);
    //b_mid
    float p5 = srcROI.at<float>((int)ceil((curH + 1) / 3.0 * 2) - 1 , 0);
    float p6 = srcROI.at<float>((int)ceil((curH + 1) / 3.0 * 2) - 1, curW);
    //bot
    float p7 = srcROI.at<float>(curH, 0);
    float p8 = srcROI.at<float>(curH, curW);
    result = (p8 + p1 - p2 - p7) - 2 * (p6 + p3 - p4 - p5);
    return result;
}

LineFeature::~LineFeature()
{
}
