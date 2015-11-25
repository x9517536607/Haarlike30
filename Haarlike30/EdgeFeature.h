#pragma once
#include "HaarFeatures.h"

class EdgeFeature : public HaarFeatures
{
    public:
        EdgeFeature(int scaleH, int scaleW, int x, int y);
        ~EdgeFeature();
        float cal(Mat ConcatImage);
        static Point getMINSize();
        int getType()
        {
            return 0;
        };
    private:

};

