#pragma once
#include "Tools.h"

class DataManager
{
public:
	DataManager(string Dpath);
	~DataManager();
	vector<Mat>* getIIGMats() { return &iigMats; }
private:
	void LoadDirectory();
	void getIIGImages();
	vector<Mat> oriMats;
	vector<Mat> iigMats;
	string directory;
};

