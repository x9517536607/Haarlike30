#pragma once
#include "Tools.h"

class StrongClassfier
{
public:
	StrongClassfier(string path, float _threshold = 0.45);
	~StrongClassfier();
	int cal(Mat &IIG);
private:
	Classifier<float>::CollectionClassifiers classifiers;
	vector<HaarFeatures *> allfeatures;
	vector<float> weights;
	float threshold;
};

