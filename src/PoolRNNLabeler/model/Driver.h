#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include <iostream>
#include "ComputionGraph.h"


//A native neural network classfier using only word embeddings

class Driver{
public:
	Driver(int memsize) :_aligned_mem(memsize){
		_pcg = NULL;
	}

	~Driver() {
		if (_pcg != NULL)
			delete _pcg;
		_pcg = NULL;
	}

public:
	ComputionGraph *_pcg;  // build neural graphs
	ModelParams _model_params;  // model parameters
	HyperParams _hyper_params;

	Metric _eval;
	CheckGrad _checkgrad;
	ModelUpdate _ada;  // model update
	AlignedMemoryPool _aligned_mem;

public:
	//embeddings are initialized before this separately.
	inline void initial() {
		if (!_hyper_params.bValid()){
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}
		if (!_model_params.initial(_hyper_params, &_aligned_mem)){
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}
		_model_params.exportModelParams(_ada);
		_model_params.exportCheckGradParams(_checkgrad);

		_hyper_params.print();

		_pcg = new ComputionGraph();
		_pcg->createNodes(ComputionGraph::max_sentence_length);
		_pcg->initial(_model_params, _hyper_params, &_aligned_mem);
		std::cout << "allocated memory: " << _aligned_mem.capacity << ", total required memory: " << _aligned_mem.required << ", perc = " << _aligned_mem.capacity*1.0 / _aligned_mem.required << std::endl;

		setUpdateParameters(_hyper_params.nnRegular, _hyper_params.adaAlpha, _hyper_params.adaEps);
	}


	inline dtype train(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			//forward
			_pcg->forward(example.m_features, example.left, example.right, true);

			//loss function
			int seq_size = example.m_features.size();
			//for (int idx = 0; idx < seq_size; idx++) {
			//cost += _loss.loss(&(_pcg->output[idx]), example.m_labels[idx], _eval, example_num);				
			//}
			cost += _model_params.loss.loss(&_pcg->_output, example.m_labels, _eval, example_num);

			// backward, which exists only for training 
			_pcg->backward();
		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	inline void predict(const vector<Feature>& features, int left, int right, int& result) {
		_pcg->forward(features, left, right);
		int seq_size = features.size();
		//results.resize(seq_size);
		//for (int idx = 0; idx < seq_size; idx++) {
		//	_loss.predict( &(_pcg->output[idx]), results[idx]);
		//}
		_model_params.loss.predict(&_pcg->_output, result);
	}

	inline dtype cost(const Example& example){
		_pcg->forward(example.m_features, example.left, example.right); //forward here

		int seq_size = example.m_features.size();

		dtype cost = 0.0;
		//loss function
		//for (int idx = 0; idx < seq_size; idx++) {
		//	cost += _loss.cost(&(_pcg->output[idx]), example.m_labels[idx], 1);
		//}
		cost += _model_params.loss.cost(&_pcg->_output, example.m_labels, 1);

		return cost;
	}


	void updateModel() {
		//_ada.update();
		_ada.update(5.0);
	}

	void checkgrad(const vector<Example>& examples, int iter){
		ostringstream out;
		out << "Iteration: " << iter;
		_checkgrad.check(this, examples, out.str());
	}

	void writeModel();

	void loadModel();



private:
	inline void resetEval() {
		_eval.reset();
	}


	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

};

#endif /* SRC_Driver_H_ */
