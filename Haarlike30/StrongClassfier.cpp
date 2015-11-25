#include "StrongClassfier.h"


StrongClassfier::StrongClassfier(string path,float _threshold)
{
	ifstream myfile(path);
	if (myfile.is_open())
	{
		string line;
		float sum = 0;
		while (getline(myfile, line))
		{
			vector<string> lineData;
			Tools::stringSplit(line, ":", &lineData);
			vector<string> pData;
			vector<string> sData;
			Tools::stringSplit(lineData[1], ",", &pData);
			Tools::stringSplit(lineData[2], ",", &sData);
			int scaleH = atoi(sData[0].c_str());
			int scaleW = atoi(sData[1].c_str());
			int x = atoi(pData[0].c_str());
			int y = atoi(pData[1].c_str());
			if (lineData[0] == "0")
			{
				EdgeFeature* F = new EdgeFeature(scaleH, scaleW, x, y);
				allfeatures.push_back(F);
			}
			else if (lineData[0] == "1")
			{
				EdgeFeature2* F = new EdgeFeature2(scaleH, scaleW, x, y);
				allfeatures.push_back(F);
			}
			else if (lineData[0] == "2")
			{
				FourRecFeature* F = new FourRecFeature(scaleH, scaleW, x, y);
				allfeatures.push_back(F);
			}
			else if (lineData[0] == "3")
			{
				LineFeature* F = new LineFeature(scaleH, scaleW, x, y);
				allfeatures.push_back(F);
			}
			else if (lineData[0] == "4")
			{
				LineFeature2* F = new LineFeature2(scaleH, scaleW, x, y);
				allfeatures.push_back(F);
			}
			float threshold = atof(lineData[4].c_str());
			int pSign = atoi(lineData[3].c_str());
			sum += atof(lineData[5].c_str());
			classifiers.push_back(new MyWeakClassifier(pSign, threshold));
			weights.push_back(atof(lineData[5].c_str()));
		}
		threshold = sum * _threshold;
		myfile.close();
	}
}


StrongClassfier::~StrongClassfier()
{
}

int StrongClassfier::cal(Mat &IIG)
{
	int featureSize = allfeatures.size();
	float val = 0;
	for (int i = 0; i < featureSize; i++)
	{
		float fv = allfeatures[i]->cal(IIG);
		int temp = classifiers[i].analyze(fv)>0 ? 1 : 0;
		val += temp*weights[i];
	}
	if (val >= threshold)return 1;
	else return 0;
}
