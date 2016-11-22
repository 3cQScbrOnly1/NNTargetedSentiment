#ifndef SRC_Model_params_H_
#define SRC_Model_params_H_

#include "HyperParams.h"

class ModelParams{
public:
	Alphabet wordAlpha;
	LookupTable words;

	UniParams represent_transform_layer;
	AttRecursiveGatedParams arg_layer;
	RNNParams rnn_layer;
	UniParams hidden_layer;
	UniParams olayer_linear;
public:
	Alphabet labelAlpha;
	Alphabet featAlpha;
	Alphabet charAlpha;

	SoftMaxLoss loss;

public:
	bool initial(HyperParams& hyper_params){
		if (words.nVSize <= 0 || labelAlpha.size() <= 0)
			return false;
		hyper_params.wordDim = words.nDim;
		hyper_params.labelSize = labelAlpha.size();
		hyper_params.wordWindow = hyper_params.wordContext * 2 + 1;
		hyper_params.windowOutputSize = hyper_params.wordDim * hyper_params.wordWindow;
		hyper_params.representInputSize = hyper_params.hiddenSize * 4;
		hyper_params.inputSize = hyper_params.hiddenSize + hyper_params.representInputSize * 3;

		rnn_layer.initial(hyper_params.rnnHiddenSize, hyper_params.windowOutputSize);
		hidden_layer.initial(hyper_params.hiddenSize, hyper_params.rnnHiddenSize * 2, true);
		represent_transform_layer.initial(hyper_params.hiddenSize, hyper_params.representInputSize, true);
		arg_layer.initial(hyper_params.hiddenSize, hyper_params.hiddenSize);
		olayer_linear.initial(hyper_params.labelSize, hyper_params.inputSize, false);
		return true;
	}

	void exportModelParams(ModelUpdate& ada) {
		words.exportAdaParams(ada);
		rnn_layer.exportAdaParams(ada);
		hidden_layer.exportAdaParams(ada);
		represent_transform_layer.exportAdaParams(ada);
		arg_layer.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}

	void exportCheckGradParams(CheckGrad& checkgrad) {
		checkgrad.add(&(rnn_layer._rnn.W1), "rnn_layer._rnn.W1");
		checkgrad.add(&(rnn_layer._rnn.W2), "rnn_layer._rnn.W2");
		checkgrad.add(&(rnn_layer._rnn.b), "rnn_layer._rnn.b");
		checkgrad.add(&(represent_transform_layer.W), "represent_transform_layer.W");
		checkgrad.add(&(arg_layer._update_left.W1), "arg_layer._update_left.W1");
		checkgrad.add(&(arg_layer._update_left.W2), "arg_layer._update_left.W2");
		checkgrad.add(&(arg_layer._update_right.W1), "arg_layer._update_right.W1");
		checkgrad.add(&(arg_layer._update_right.W2), "arg_layer._update_right.W2");
		checkgrad.add(&(arg_layer._reset_left.W1), "arg_layer._reset_left.W1");
		checkgrad.add(&(arg_layer._reset_left.W2), "arg_layer._reset_left.W2");
		checkgrad.add(&(arg_layer._reset_right.W1), "arg_layer._reset_right.W1");
		checkgrad.add(&(arg_layer._reset_right.W2), "arg_layer._reset_right.W2");
		checkgrad.add(&(arg_layer._update_tilde.W1), "arg_layer._update_tilde.W1");
		checkgrad.add(&(arg_layer._update_tilde.W2), "arg_layer._update_tilde.W2");
		checkgrad.add(&(arg_layer._recursive_tilde.W1), "arg_layer._recursive_tilde.W1");
		checkgrad.add(&(arg_layer._recursive_tilde.W2), "arg_layer._recursive_tilde.W2");
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
	}

	void saveModel(){
	}

	void loadModel(const string& infile) {
	}
};

#endif
