#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"

struct ComputionGraph :Graph{
public:
	const static int max_sentence_length = 256;
private:
	int padding_dim;

public:
	vector<LookupNode> _word_inputs;
	WindowBuilder _word_window;
	GRNNBuilder _grnn_left;
	GRNNBuilder _grnn_right;
	vector<ConcatNode> _bi_rnn_concat;
	vector<UniNode> _hidden;

	MaxPoolNode _left_max_pooling;
	MinPoolNode _left_min_pooling;
	AvgPoolNode _left_avg_pooling;
	StdPoolNode _left_std_pooling;
	ConcatNode _left_pooling_concat;

	MaxPoolNode _target_max_pooling;
	MinPoolNode _target_min_pooling;
	AvgPoolNode _target_avg_pooling;
	StdPoolNode _target_std_pooling;
	ConcatNode _target_pooling_concat;

	MaxPoolNode _right_max_pooling;
	MinPoolNode _right_min_pooling;
	AvgPoolNode _right_avg_pooling;
	StdPoolNode _right_std_pooling;
	ConcatNode _right_pooling_concat;
	Node _pooling_concat_zero;

	ConcatNode _concat;
	LinearNode _output;
public:
	ComputionGraph():Graph(){
	}

	~ComputionGraph(){
		clear();
	}
public:
	inline void createNodes(int sent_length) {
		_word_inputs.resize(sent_length);
		_word_window.resize(sent_length);
		_grnn_left.resize(sent_length);
		_grnn_right.resize(sent_length);
		_bi_rnn_concat.resize(sent_length);
		_hidden.resize(sent_length);
	}

	inline void clear(){
		_word_inputs.clear();
	}

public:
	inline void initial(ModelParams& model_params, HyperParams& hyper_params) {
		for (int idx = 0; idx < _word_inputs.size(); idx++)
			_word_inputs[idx].setParam(&model_params.words);
		_pooling_concat_zero.val =  Mat::Zero(hyper_params.hiddenSize * 4, 1);
		_word_window.setContext(hyper_params.wordContext);
		_grnn_left.setParam(&model_params.grnn_layer, hyper_params.dropOut, true);
		_grnn_right.setParam(&model_params.grnn_layer, hyper_params.dropOut, false);
		for (int idx = 0; idx < _hidden.size(); idx++)
			_hidden[idx].setParam(&model_params.hidden_layer);
		_output.setParam(&model_params.olayer_linear);
	}

public:
	inline void forward(const vector<Feature>& features, int left, int right, bool bTrain = false){
		clearValue(bTrain);
		int seq_size = features.size();
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			_word_inputs[idx].forward(this, feature.words[0]);
		}
		_word_window.forward(this, getPNodes(_word_inputs, seq_size));
		_grnn_left.forward(this, getPNodes(_word_window._outputs, seq_size));
		_grnn_right.forward(this, getPNodes(_word_window._outputs, seq_size));
		for (int idx = 0; idx < seq_size; idx++)
		{
			_bi_rnn_concat[idx].forward(this, &_grnn_left._output[idx], &_grnn_right._output[idx]);
			_hidden[idx].forward(this, &_bi_rnn_concat[idx]);
		}
		vector<PNode> three_part_poolings;
		if (left != 0)
		{
			_left_max_pooling.forward(this, getPNodes(_hidden, left));
			_left_min_pooling.forward(this, getPNodes(_hidden, left));
			_left_avg_pooling.forward(this, getPNodes(_hidden, left));
			_left_std_pooling.forward(this, getPNodes(_hidden, left));
			_left_pooling_concat.forward(this, &_left_max_pooling, &_left_min_pooling, &_left_avg_pooling, &_left_std_pooling);
			three_part_poolings.push_back(&_left_pooling_concat);
		} else
			three_part_poolings.push_back(&_pooling_concat_zero);
		if (right != 0)
		{
			_target_max_pooling.forward(this, getPNodes(_hidden, left, right - left + 1));
			_target_min_pooling.forward(this, getPNodes(_hidden, left, right - left + 1));
			_target_avg_pooling.forward(this, getPNodes(_hidden, left, right - left + 1));
			_target_std_pooling.forward(this, getPNodes(_hidden, left, right - left + 1));
			_target_pooling_concat.forward(this, &_target_max_pooling, &_target_min_pooling, &_target_avg_pooling, &_target_std_pooling);
			three_part_poolings.push_back(&_target_pooling_concat);
		} else
			three_part_poolings.push_back(&_pooling_concat_zero);
		if (right != seq_size - 1)
		{
			_right_max_pooling.forward(this, getPNodes(_hidden, right + 1, seq_size - right - 1));
			_right_min_pooling.forward(this, getPNodes(_hidden, right + 1, seq_size - right - 1));
			_right_avg_pooling.forward(this, getPNodes(_hidden, right + 1, seq_size - right - 1));
			_right_std_pooling.forward(this, getPNodes(_hidden, right + 1, seq_size - right - 1));
			_right_pooling_concat.forward(this, &_right_max_pooling, &_right_min_pooling, &_right_avg_pooling, &_right_std_pooling);
			three_part_poolings.push_back(&_right_pooling_concat);
		} else
			three_part_poolings.push_back(&_pooling_concat_zero);
		_concat.forward(this, three_part_poolings);
		_output.forward(this, &_concat);
	}
};

#endif