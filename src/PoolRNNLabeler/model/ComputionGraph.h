#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"

struct ComputionGraph :Graph{
public:
	const static int max_sentence_length = 256;
public:
	vector<LookupNode> _word_inputs1;
	vector<LookupNode> _word_inputs2;
	vector<ConcatNode> _word_concats;

	WindowBuilder _word_window;
	RNNBuilder _rnn_left;
	RNNBuilder _rnn_right;
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
		_word_inputs1.resize(sent_length);
		_word_inputs2.resize(sent_length);
		_word_concats.resize(sent_length);
		_word_window.resize(sent_length);
		_rnn_left.resize(sent_length);
		_rnn_right.resize(sent_length);
		_bi_rnn_concat.resize(sent_length);

		_left_max_pooling.setParam(sent_length);
		_left_min_pooling.setParam(sent_length);
		_left_avg_pooling.setParam(sent_length);
		_left_std_pooling.setParam(sent_length);

		_right_max_pooling.setParam(sent_length);
		_right_min_pooling.setParam(sent_length);
		_right_avg_pooling.setParam(sent_length);
		_right_std_pooling.setParam(sent_length);

		_target_max_pooling.setParam(sent_length);
		_target_min_pooling.setParam(sent_length);
		_target_avg_pooling.setParam(sent_length);
		_target_std_pooling.setParam(sent_length);

		_hidden.resize(sent_length);
	}

	inline void clear(){
		_word_inputs1.clear();
		_word_inputs2.clear();
		_word_concats.clear();
		_word_window.clear();
		_rnn_left.clear();
		_rnn_right.clear();
		_bi_rnn_concat.clear();
		_hidden.clear();
	}

public:
	inline void initial(ModelParams& model_params, HyperParams& hyper_params, AlignedMemoryPool *mem = NULL) {
		int maxsize = _word_inputs1.size();
		for (int idx = 0; idx < maxsize; idx++)
		{
			_word_inputs1[idx].setParam(&model_params.words1);
			_word_inputs1[idx].init(hyper_params.wordDim1, hyper_params.dropOut, mem);
			_word_inputs2[idx].setParam(&model_params.words2);
			_word_inputs2[idx].init(hyper_params.wordDim2, hyper_params.dropOut, mem);

			_word_concats[idx].init(hyper_params.wordDim, -1, mem);

			_bi_rnn_concat[idx].init(hyper_params.rnnHiddenSize * 2, hyper_params.dropOut, mem);
		}
		_pooling_concat_zero.init(hyper_params.hiddenSize * 4, -1, mem);
		_word_window.init(hyper_params.wordDim, hyper_params.wordContext, mem);
		_rnn_left.init(&model_params.rnn_left_layer, hyper_params.dropOut, true, mem);
		_rnn_right.init(&model_params.rnn_right_layer, hyper_params.dropOut, false, mem);
		for (int idx = 0; idx < _hidden.size(); idx++)
		{
			_hidden[idx].setParam(&model_params.hidden_layer);
			_hidden[idx].init(hyper_params.hiddenSize, hyper_params.dropOut, mem);
		}
		_left_max_pooling.init(hyper_params.hiddenSize, -1, mem);
		_left_min_pooling.init(hyper_params.hiddenSize, -1, mem);
		_left_avg_pooling.init(hyper_params.hiddenSize, -1, mem);
		_left_std_pooling.init(hyper_params.hiddenSize, -1, mem);

		_left_pooling_concat.init(hyper_params.hiddenSize * 4, -1, mem);

		_right_max_pooling.init(hyper_params.hiddenSize, -1, mem);
		_right_min_pooling.init(hyper_params.hiddenSize, -1, mem);
		_right_avg_pooling.init(hyper_params.hiddenSize, -1, mem);
		_right_std_pooling.init(hyper_params.hiddenSize, -1, mem);

		_right_pooling_concat.init(hyper_params.hiddenSize * 4, -1, mem);

		_target_max_pooling.init(hyper_params.hiddenSize, -1, mem);
		_target_min_pooling.init(hyper_params.hiddenSize, -1, mem);
		_target_avg_pooling.init(hyper_params.hiddenSize, -1, mem);
		_target_std_pooling.init(hyper_params.hiddenSize, -1, mem);

		_target_pooling_concat.init(hyper_params.hiddenSize * 4, -1, mem);

		_concat.init(hyper_params.hiddenSize * 4 * 3, -1, mem);

		_output.setParam(&model_params.olayer_linear);
		_output.init(hyper_params.labelSize, hyper_params.dropOut, mem);
	}

public:
	inline void forward(const vector<Feature>& features, int left, int right, bool bTrain = false){
		clearValue(bTrain);
		int seq_size = features.size();
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			_word_inputs1[idx].forward(this, feature.words[0]);
			_word_inputs2[idx].forward(this, feature.words[0]);
			_word_concats[idx].forward(this, &_word_inputs1[idx], &_word_inputs2[idx]);
		}
		_word_window.forward(this, getPNodes(_word_concats, seq_size));
		_rnn_left.forward(this, getPNodes(_word_window._outputs, seq_size));
		_rnn_right.forward(this, getPNodes(_word_window._outputs, seq_size));
		for (int idx = 0; idx < seq_size; idx++)
		{
			_bi_rnn_concat[idx].forward(this, &_rnn_left._output[idx], &_rnn_right._output[idx]);
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