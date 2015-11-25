#include "HaarFeatures.h"


HaarFeatures::HaarFeatures()
{
}

float HaarFeatures::getResult()
{
    return result;
}

HaarFeatures::~HaarFeatures()
{

}

Size HaarFeatures::getSize()
{
    return Size(minW * scaleW, minH * scaleH);
}

Point HaarFeatures::getPosition()
{
    return Point(x, y);
}

Point HaarFeatures::getMINSize()
{
    return Point(0, 0);
}
