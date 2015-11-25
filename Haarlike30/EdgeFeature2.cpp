#include "EdgeFeature2.h"

EdgeFeature2::EdgeFeature2(int scaleH, int scaleW, int x, int y)
{
    this->scaleH = scaleH;
    this->scaleW = scaleW;
    this->x = x;
    this->y = y;
    this->minW = 1;
    this->minH = 2;
}

Point EdgeFeature2::getMINSize()
{
    return Point(1, 2);
}

EdgeFeature2::~EdgeFeature2()
{
}

float EdgeFeature2::cal(Mat ConcatImage)
{
    /*
    ¢z¢w¢s¢w¢{
    ¢x1¢x2¢x
    ¢u¢w¢q¢w¢t
    ¢x6¢x3¢x
    ¢u¢w¢q¢w¢t
    ¢x5¢x4¢x
    ¢|¢w¢r¢w¢}
    */
    int curH = minH * scaleH;
    int curW = minW * scaleW;
    /*if (integralImage.cols < x + curW || integralImage.rows < y + curH || curH == 0 || curW == 0)
        return FLT_MIN;*/
    //Mat haar = Mat(curH, curW, CV_32F);
    //Tools::printImage(integralImage);
    //Mat ConcatImage = Tools::getConcatImage(integralImage);
    Mat srcROI(ConcatImage, Rect(x, y, curW + 1, curH + 1));
    //Tools::printImage(srcROI);
    float p1 = srcROI.at<float>(0, 0);
    float p2 = srcROI.at<float>(0, curW );
    float p3 = srcROI.at<float>((int)ceil((curH + 1) / 2.0) - 1, curW);
    float p4 = srcROI.at<float>(curH , curW);
    float p5 = srcROI.at<float>(curH , 0);
    float p6 = srcROI.at<float>((int)ceil((curH + 1 ) / 2.0) - 1, 0);
    result = (p3 - p6 - p2 + p1) - (p4 - p5 - p3 + p6);
    return result;
}