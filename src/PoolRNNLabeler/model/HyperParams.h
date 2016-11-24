#ifndef SRC_Hyperparams_H_
#define SRC_Hyperparams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{
	int rnnHiddenSize;
	int hiddenSize;
	int wordContext;
	dtype dropOut;

	dtype nnRegular;
	dtype adaAlpha;
	dtype adaEps;

	int wordDim1;
	int wordDim2;
	int wordDim;
	int wordWindow;
	int windowOutputSize;
	int inputSize;
	int labelSize;
	

	HyperParams(){
		bAssigned = false;
	}

	void setRequared(Options& opt) {
		wordContext = opt.wordcontext;
		hiddenSize = opt.hiddenSize;
		rnnHiddenSize = opt.rnnHiddenSize; 
		dropOut = opt.dropProb;
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;

		bAssigned = true;
	}

	void print(){
	}

	void clear() {
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}

private:
	bool bAssigned;
};

#endif
