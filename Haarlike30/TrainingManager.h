#pragma once
#include "DataManager.h"

class TrainingManager
{
public:
	TrainingManager(DataManager &posData, DataManager &negData);
	TrainingManager(vector<Mat> &posIIGs, vector<Mat> &negIIGs);
	~TrainingManager();
	vector<HaarValue>* getAllfeature() { return &allfeature; };
	vector<vector<int>>* getRM() { return &RM; };
	Labels* getLabels() { return &labels; };
private:
	template<class T>
	void getAllHaarFeature(const vector<Mat>& posIIG, const vector<Mat>& negIIG);
	float getVectorMean(vector<float>& h);
	void getResultMatrix(Classifier<float> *wc, HaarValue &haarValue);
	void init(vector<Mat> &posIIGs, vector<Mat> &negIIGs);

	vector<HaarValue> allfeature;
	vector<vector<int>> RM;
	Labels labels;
};

