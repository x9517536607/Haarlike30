#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

class HaarFeatures
{
    public:
        HaarFeatures();
        float getResult();
        Size getSize();
        Point getPosition();
        static Point getMINSize();
		virtual int getType() = 0;
		virtual ~HaarFeatures() = 0;
		virtual float cal(Mat integralImage) = 0;

    protected:
        int scaleH;
        int scaleW;
        int x;
        int y;
        int minW;
        int minH;
        float result;
};

