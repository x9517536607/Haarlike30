#pragma once
#include "HaarFeatures.h"

class EdgeFeature2 : public HaarFeatures
{
    public:
        EdgeFeature2(int scaleH, int scaleW, int x, int y);
        ~EdgeFeature2();
        float cal(Mat ConcatImage);
        static Point getMINSize();
        int getType()
        {
            return 1;
        };
};

