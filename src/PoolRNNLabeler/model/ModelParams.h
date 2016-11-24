#ifndef SRC_Model_params_H_
#define SRC_Model_params_H_

#include "HyperParams.h"

class ModelParams{
public:
	LookupTable words1;
	LookupTable words2;

	RNNParams rnn_layer;
	UniParams hidden_layer;
	UniParams olayer_linear;
public:
	Alphabet wordAlpha;
	Alphabet labelAlpha;
	Alphabet featAlpha;
	Alphabet charAlpha;

	SoftMaxLoss loss;

public:
	bool initial(HyperParams& hyper_params){
		if (words1.nVSize <= 0 || labelAlpha.size() <= 0)
			return false;
		hyper_params.wordDim1 = words1.nDim;
		hyper_params.wordDim2 = words2.nDim;
		hyper_params.wordDim = words1.nDim + words2.nDim;
		hyper_params.labelSize = labelAlpha.size();
		hyper_params.wordWindow = hyper_params.wordContext * 2 + 1;
		hyper_params.windowOutputSize = hyper_params.wordDim * hyper_params.wordWindow;
		hyper_params.inputSize = hyper_params.hiddenSize * 3 * 4;

		rnn_layer.initial(hyper_params.rnnHiddenSize, hyper_params.windowOutputSize);
		hidden_layer.initial(hyper_params.hiddenSize, hyper_params.rnnHiddenSize * 2, true);
		olayer_linear.initial(hyper_params.labelSize, hyper_params.inputSize, false);
		return true;
	}

	void exportModelParams(ModelUpdate& ada) {
		words1.exportAdaParams(ada);
		words2.exportAdaParams(ada);
		rnn_layer.exportAdaParams(ada);
		hidden_layer.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}

	void exportCheckGradParams(CheckGrad& checkgrad) {
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
	}

	void saveModel(){
	}

	void loadModel(const string& infile) {
	}
};

#endif
