#ifndef WEAK_CLASSIFIER
#define WEAK_CLASSIFIER 1
#include "adaboost.hpp"

namespace DM_AG
{
class MyWeakClassifier : public Classifier<float>
{
    public:
        MyWeakClassifier(int _p, float _threshold)
        {
            p = _p;
            threshold = _threshold;
        }

        int analyze(const float& feature) const
        {
            if (feature * p <= threshold * p) return 1;
            return -1;
        }
};

};
#endif
