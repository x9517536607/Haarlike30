#include "TrainingManager.h"



TrainingManager::TrainingManager(DataManager &posData, DataManager &negData)
{
	init(*(posData.getIIGMats()), *(negData.getIIGMats()));
}

TrainingManager::TrainingManager(vector<Mat> &posIIGs, vector<Mat> &negIIGs) 
{
	init(posIIGs, negIIGs);
}

void TrainingManager::init(vector<Mat> &posIIGs, vector<Mat> &negIIGs)
{
	getAllHaarFeature<EdgeFeature>(posIIGs, negIIGs);
	getAllHaarFeature<EdgeFeature2>(posIIGs, negIIGs);
	getAllHaarFeature<LineFeature>(posIIGs, negIIGs);
	getAllHaarFeature<LineFeature2>(posIIGs, negIIGs);
	getAllHaarFeature<FourRecFeature>(posIIGs, negIIGs);
	int featureSize = allfeature.size();
	if (featureSize > 0)
	{
		for (int i = 0; i < allfeature[0].POSFVs.size(); i++)
		{
			labels.push_back(1);
		}
		for (int i = 0; i < allfeature[0].NEGFVs.size(); i++)
		{
			labels.push_back(-1);
		}
	}
	for (int i = 0; i < featureSize; i++)
	{
		float pos = getVectorMean(allfeature[i].POSFVs);
		float neg = getVectorMean(allfeature[i].NEGFVs);
		allfeature[i].posThreshold = pos;
		allfeature[i].negThreshold = neg;
		allfeature[i].pSign = (pos < neg) ? 1 : -1;
		MyWeakClassifier *wc = new MyWeakClassifier(allfeature[i].pSign, pos);
		getResultMatrix(wc, allfeature[i]);
	}
}

void TrainingManager::getResultMatrix(Classifier<float> *wc, HaarValue &haarValue)
{
	int posSize = haarValue.POSFVs.size();
	int negSize = haarValue.NEGFVs.size();
	vector<int> result;
	for (int i = 0; i < posSize; i++)
	{
		result.push_back((*wc).analyze(haarValue.POSFVs[i]));
	}
	for (int i = 0; i < negSize; i++)
	{
		result.push_back((*wc).analyze(haarValue.NEGFVs[i]));
	}
	RM.push_back(result);
}

float TrainingManager::getVectorMean(vector<float>& h)
{
	int featureSize = h.size();
	float sum = 0;
	for (int i = 0; i < featureSize; i++)
	{
		sum += h[i];
	}
	return sum / (float)featureSize;
}

TrainingManager::~TrainingManager()
{
}

template<class T>
void TrainingManager::getAllHaarFeature( const vector<Mat>& posIIG, const vector<Mat>& negIIG)
{
	int simpleH = 24;
	int simpleW = 24;
	int minH = T::getMINSize().y;
	int minW = T::getMINSize().x;
	int maxScaleH = simpleH / minH;
	int maxScaleW = simpleW / minW;
	int posCount = posIIG.size();
	int negCount = negIIG.size();
	for (int h = 1; h <= maxScaleH; h++)
	{
		for (int w = 1; w <= maxScaleW; w++)
		{
			for (int y = 0; y < simpleH - minH * h; y++)
			{
				for (int x = 0; x < simpleW - minW * w; x++)
				{
					T e = T(h, w, x, y);
					HaarValue haarValue(e.getType(), Size(w, h), Point(x, y));
					for (int i = 0; i < posCount; i++)
					{
						haarValue.POSFVs.push_back(e.cal(posIIG[i]));
					}
					for (int i = 0; i < negCount; i++)
					{
						haarValue.NEGFVs.push_back(e.cal(negIIG[i]));
					}
					allfeature.push_back(haarValue);
				}
			}
		}
	}
}
