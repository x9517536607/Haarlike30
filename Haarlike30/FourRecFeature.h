#pragma once
#include "HaarFeatures.h"

class FourRecFeature : public HaarFeatures
{
    public:
        FourRecFeature(int scaleH, int scaleW, int x, int y);
        ~FourRecFeature();
        float cal(Mat ConcatImage);
        static Point getMINSize();
        int getType()
        {
            return 2;
        };
};

