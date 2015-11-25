#include "LineFeature2.h"


LineFeature2::LineFeature2(int scaleH, int scaleW, int x, int y)
{
    this->scaleH = scaleH;
    this->scaleW = scaleW;
    this->x = x;
    this->y = y;
    this->minW = 3;
    this->minH = 1;
}

Point LineFeature2::getMINSize()
{
    return Point(3, 1);
}

float LineFeature2::cal(Mat ConcatImage)
{
    /*
    ¢z¢w¢s¢w¢s¢w¢s¢w¢{
    ¢x1¢x2¢x3¢x4¢x
    ¢u¢w¢q¢w¢q¢w¢q¢w¢t
    ¢x5¢x6¢x7¢x8¢x
    ¢|¢w¢r¢w¢r¢w¢r¢w¢}
       + - +
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
    float p2 = srcROI.at<float>(0, (int)ceil((curW + 1) / 3.0) - 1);
    float p3 = srcROI.at<float>(0, (int)ceil((curW + 1) / 3.0 * 2) - 1);
    float p4 = srcROI.at<float>(0, curW);
    //bot
    float p5 = srcROI.at<float>(curH, 0);
    float p6 = srcROI.at<float>(curH, (int)ceil((curW + 1) / 3.0) - 1);
    float p7 = srcROI.at<float>(curH, (int)ceil((curW + 1) / 3.0 * 2) - 1);
    float p8 = srcROI.at<float>(curH, curW);
    result = (p8 + p1 - p4 - p5) - 2 * (p7 + p2 - p3 - p6);
    return result;
}

LineFeature2::~LineFeature2()
{
}
