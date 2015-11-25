#ifndef ADA_BOOST
#define ADA_BOOST 1
#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>

namespace DM_AG
{

typedef std::vector<float> ClassificationResults;
typedef std::vector<int> Labels;

// A classifier
//    abstract class

template <typename T>
class Classifier
{

    public:
        typedef typename boost::ptr_vector<Classifier<T> > CollectionClassifiers;
        typedef typename std::vector<T> Data;
        int p = 1;
        float threshold = 0;
        virtual int analyze(const T& feature) const = 0;
};

template <typename T>
class ADA
{

    public:

        //
        // Apply Adaboost
        //
        //  @param weak_classifiers, a set of weak classifiers
        //  @param data, the dataset to classify
        //  @param labels, classification labels (e.g. -1; +1}
        //  @param num_rounds, # boost iteration (default 100)

        ClassificationResults
        ada_boost(int classifiers_size,
			std::vector<std::vector<int>> &MR,
			const Labels& labels, 
			const unsigned int num_iterations)
        {
            // following notation
            //
            // http://en.wikipedia.org/wiki/AdaBoost
            ClassificationResults alpha;
            ClassificationResults D;
            size_t labels_size = labels.size();
            D.resize(labels_size);           // D
            alpha.resize(classifiers_size);       // alpha
            unsigned int num_current_classifier = 0;
            // Init boosters
            //for (unsigned int j = 0; j < labels_size; j++)
            //    D[j] = (1.0) / labels_size;   // init D


			int posCount = 0;
			int negCount = 0;
			for (unsigned int j = 0; j < labels_size; j++)
			{
				if (labels[j] == 1)
					posCount++;
				else
					negCount++;
			}
			float z = 0;
			for (unsigned int j = 0; j < labels_size; j++)
			{
				if (labels[j] == 1)
					D[j] = (1.0) / (2.0*posCount);
				else
					D[j] = (1.0) / (2.0*negCount);
				z += D[j];

			}
			for (unsigned int j = 0; j < labels_size; j++)
				D[j] /= z;
            // for the maximum rounds
            //
            for (unsigned int round = 0;
                    round < num_iterations; round++)
            {
                //std::cout << "Iteration" << round << std::endl;
                float min_error = labels_size;
                unsigned int best_classifier = 0;
                //
                // for each classifier
                for (num_current_classifier = 0;
                        num_current_classifier < classifiers_size;
                        num_current_classifier++)
                {
                    float error = 0;
                    //
                    // for each feature
                    for (unsigned int j = 0; j < labels_size; j++)
                        if (MR[num_current_classifier][j] != labels[j])
                            error += D[j];
                    if (error < min_error)
                    {
                        min_error = error; // this is the best observed
                        best_classifier = num_current_classifier;
                    }
                }// each classifier
                //std::cout << "\tbest_classifier=" << best_classifier
                 //         << " error=" << min_error << std::endl;
                if (min_error >= 0.5)    // GOOD enough
                    break;                 // condition
                // a_t
                alpha[best_classifier] =
                    log((1.0f - min_error) / min_error) / 2;
                // D_{t+1}
                ClassificationResults D_1(D);
                // update D_{t+1}
                float z = 0;
                for (unsigned int j = 0; j < labels_size; j++)
                {
                    D_1[j] *=
                        exp(-alpha[best_classifier] *
                            labels[j] *
							MR[best_classifier][j]);
                    z += D_1[j];
                }
                // normalize so that it is a prob distribution
                for (unsigned int j = 0; j < labels_size; j++)
                    D[j] = D_1[j] / z;
            } // all the rounds.
            return alpha;
        };

}; // class ADA

} // namespace

#endif
