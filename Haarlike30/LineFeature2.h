#pragma once
#include "HaarFeatures.h"

class LineFeature2 : public HaarFeatures
{
    public:
        LineFeature2(int scaleH, int scaleW, int x, int y);
        ~LineFeature2();
        float cal(Mat ConcatImage);
        static Point getMINSize();
        int getType()
        {
            return 4;
        };
};

