#pragma once
#include "HaarFeatures.h"

class LineFeature : public HaarFeatures
{
    public:
        LineFeature(int scaleH, int scaleW, int x, int y);
        ~LineFeature();
        float cal(Mat ConcatImage);
        static Point getMINSize();
        int getType()
        {
            return 3;
        };
};

