#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "adaBoost/adaboost.hpp"
#include "Tools.h"
#include <string>
#include <time.h>
#include "StrongClassfier.h"
#include "DataManager.h"
#include "TrainingManager.h"

using namespace std;
using namespace cv;
using namespace DM_AG;

void saveStrongClassfier(vector<HaarValue> &allFeature, ClassificationResults &weights);
void train(vector<HaarValue> &allfeature, vector<Mat> &posIIGs, vector<Mat> &negIIGs);
void K_fold_cross_validation(int k, DataManager &posData, DataManager &negData);

int main()
{
	time_t start;
    start = clock();
	DataManager posData("E:\\論文相關\\孝民論文程式_v1\\NewTrain\\PDSTrain\\PDSTrain\\Train\\CascadeTrainPos_car");
	DataManager negData("E:\\論文相關\\孝民論文程式_v1\\NewTrain\\PDSTrain\\PDSTrain\\Train\\Neg");
	K_fold_cross_validation(4, posData, negData);
	/*TrainingManager tm(posData, negData);
	cerr << (clock() - start) / 1000.0 << "s" << endl;
	start = clock();
	ADA<float> ada;
	std::cerr << "Boosting ... " << std::endl;
	ClassificationResults weights = ada.ada_boost(tm.getAllfeature()->size(), *(tm.getRM()), *(tm.getLabels()), 100);
	std::cerr << "Done " << std::endl;
	cerr << (clock() - start) / 1000.0 << "s" << endl;
	start = clock();
	std::cerr << "Construct model ... " << std::endl;
	saveStrongClassfier(*(tm.getAllfeature()), weights);
	cerr << (clock() - start) / 1000.0 << "s" << endl;*/
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	//
	vector<Mat> posMats = Tools::LoadDirectory("E:\\論文相關\\孝民論文程式_v1\\NewTrain\\PDSTrain\\PDSTrain\\Train\\CascadeTrainPos_car");
	vector<Mat> negMats = Tools::LoadDirectory("E:\\論文相關\\孝民論文程式_v1\\NewTrain\\PDSTrain\\PDSTrain\\Train\\Neg");
	vector<Mat> posIIGs = Tools::getIIGImages(posMats);
	vector<Mat> negIIGs = Tools::getIIGImages(negMats);
	posMats.clear();
	negMats.clear();
	cerr << (clock() - start) / 1000.0 << "s" << endl;
	//K_fold_cross_validation(4, posIIGs, negIIGs);
	//vector<HaarValue> allfeature;
	//train(allfeature, posIIGs, negIIGs);
	////
 //   int featureSize = allfeature.size();
	//vector<vector<int>> RM;
	//Labels labels;
	////
	//for (int i = 0; i < allfeature[0].POSFVs.size(); i++) 
	//{
	//	labels.push_back(1);
	//}
	////
	//for (int i = 0; i < allfeature[0].NEGFVs.size(); i++)
	//{
	//	labels.push_back(-1);
	//}
	////
 //   for (int i = 0; i < featureSize; i++)
 //   {
 //       float pos = Tools::getVectorMean(allfeature[i].POSFVs);
 //       float neg = Tools::getVectorMean(allfeature[i].NEGFVs);
 //       allfeature[i].posThreshold = pos;
 //       allfeature[i].negThreshold = neg;
 //       allfeature[i].pSign = (pos < neg) ? 1 : -1;
	//	MyWeakClassifier *wc = new MyWeakClassifier(allfeature[i].pSign, pos);
	//	Tools::getResultMatrix(wc, allfeature[i], RM);
	//	allfeature[i].NEGFVs.clear();
	//	allfeature[i].POSFVs.clear();
 //   }
	//posMats.clear();
	//negMats.clear();
	//posIIGs.clear();
	//negIIGs.clear();
 //   cerr << (clock() - start) / 1000.0 << "s" << endl;
 //   start = clock();
 //   //
 //   ADA<float> ada;
 //   std::cout << "Boosting ... " << std::endl;
 //   ClassificationResults weights = ada.ada_boost(featureSize,RM, labels, 100);
 //   std::cout << "Done " << std::endl;
 //   cerr << (clock() - start) / 1000.0 << "s" << endl;
	//saveStrongClassfier(allfeature, weights);
 //   allfeature.clear();
	getchar();
    return 0;
}

void testing(vector<Mat> &postestIIGs, vector<Mat> &negtestIIGs, int &Tp, int &Fp, int &Fn , int &Tn )
{
	int _tp = 0, _fp = 0, _fn = 0, _tn = 0;
	StrongClassfier sc("classfier.txt");
	for (int i = 0; i < postestIIGs.size(); i++)
	{
		//cerr << sc.cal(posIIGs[i]) << endl;
		if (sc.cal(postestIIGs[i]) == 1)
			_tp++;
		else
			_fn++;
		//Tools::printImage(posIIGs[i]);
	}
	for (int i = 0; i < negtestIIGs.size(); i++)
	{
		//cerr << sc.cal(negIIGs[i]) << endl;
		if (sc.cal(negtestIIGs[i]) == 1)
			_fp++;
		else
			_tn++;
	}
	Tp += _tp;
	Tn += _tn;
	Fn += _fn;
	Fp += _fp;
	cerr << "acc:" << (float)(_tp + _tn) / (_tp + _fn + _fp + _tn) << endl;
	cerr << "tpr:" << (float)_tp / (_tp + _fn) << endl;
	cerr << "fpr:" << (float)_fp / (_tn + _fp) << endl;
}

void K_fold_cross_validation(int k, DataManager &posData, DataManager &negData)
{
	time_t start;
	int Tp = 0, Fp = 0, Fn = 0, Tn = 0;
	vector<Mat> * posIIGs = posData.getIIGMats();
	vector<Mat> * negIIGs = negData.getIIGMats();
	vector<vector<Mat>> subPosIIGs(k);
	vector<vector<Mat>> subNegIIGs(k);
	for (int i = 0; i < posIIGs->size(); i++)
	{
		subPosIIGs[i % k].push_back((*posIIGs)[i]);
	}

	for (int i = 0; i < negIIGs->size(); i++)
	{
		subNegIIGs[i % k].push_back((*negIIGs)[i]);
	}

	posIIGs->clear();
	negIIGs->clear();
	for (int l = 0; l < k; l++)
	{
		std::cerr << "K = " << l << std::endl;
		std::cerr << "Training ... " << std::endl;
		start = clock();
		vector<Mat> trainPos;
		vector<Mat> trainNeg;
		for (int j = 0; j < k; j++)
		{
			if (l != j)
			{
				trainPos.insert(trainPos.end(), subPosIIGs[j].begin(), subPosIIGs[j].end());
				trainNeg.insert(trainNeg.end(), subNegIIGs[j].begin(), subNegIIGs[j].end());
			}
		}
		TrainingManager tm(trainPos, trainNeg);
		cerr << (clock() - start) / 1000.0 << "s" << endl;
		start = clock();
		//
		ADA<float> ada;
		std::cerr << "Boosting ... " << std::endl;
		ClassificationResults weights = ada.ada_boost(tm.getAllfeature()->size(), *(tm.getRM()), *(tm.getLabels()), 100);
		std::cerr << "Done " << std::endl;
		cerr << (clock() - start) / 1000.0 << "s" << endl;
		start = clock();
		std::cerr << "Construct model ... " << std::endl;
		saveStrongClassfier(*(tm.getAllfeature()), weights);
		cerr << (clock() - start) / 1000.0 << "s" << endl;
		start = clock();
		std::cerr << "Testing ... " << std::endl;
		//allfeature.clear();
		testing(subPosIIGs[l], subNegIIGs[l], Tp, Fp, Fn, Tn);
		cerr << (clock() - start) / 1000.0 << "s" << endl;
		cerr << endl;

	}
	cerr << "Final acc:" << (float)(Tp + Tn) / (Tp + Fn + Fp + Tn) << endl;
	cerr << "Final tpr:" << (float)Tp / (Tp + Fn) << endl;
	cerr << "Final fpr:" << (float)Fp / (Tn + Fp) << endl;
	cerr << endl;
}

void train(vector<HaarValue> &allfeature, vector<Mat> &posIIGs, vector<Mat> &negIIGs)
{
	Tools::getAllHaarFeature<EdgeFeature>(allfeature, posIIGs, negIIGs);
	Tools::getAllHaarFeature<EdgeFeature2>(allfeature, posIIGs, negIIGs);
	Tools::getAllHaarFeature<LineFeature>(allfeature, posIIGs, negIIGs);
	Tools::getAllHaarFeature<LineFeature2>(allfeature, posIIGs, negIIGs);
	Tools::getAllHaarFeature<FourRecFeature>(allfeature, posIIGs, negIIGs);
}

void saveStrongClassfier(vector<HaarValue> &allFeature, ClassificationResults &weights)
{
	ofstream myfile;
	myfile.open("classfier.txt");
	int featureSize = allFeature.size();
	for (unsigned int i = 0; i < featureSize; i++)
	{
		if (weights[i] != 0)
		{
			//std::cout << "w" << i << "=" << weights[i] << endl;
			myfile << allFeature[i].type << ":"//0
				<< allFeature[i].p.x << "," << allFeature[i].p.y << ":"//1
				<< allFeature[i].s.height << "," << allFeature[i].s.width << ":"//2
				<< allFeature[i].pSign << ":"//3
				<< allFeature[i].posThreshold << ":"//4
				<< weights[i] << ":"//5
				<< endl;
		}
	}
	myfile.close();
}