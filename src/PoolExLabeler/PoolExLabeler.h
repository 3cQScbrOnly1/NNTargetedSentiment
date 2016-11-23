/*
 * Labeler.h
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#ifndef SRC_NNCRF_H_
#define SRC_NNCRF_H_


#include "N3L.h"
#include "Driver.h"
#include "Options.h"
#include "Instance.h"
#include "Example.h"


#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Labeler {

public:
  unordered_map<string, int> feature_stat;
  unordered_map<string, int> word_stat;
  unordered_map<string, int> char_stat;
public:
  Options m_options;
  Pipe m_pipe;

  Driver m_driver;

public:
  Labeler(int memsize);
  virtual ~Labeler();

public:

  int createAlphabet(const vector<Instance>& vecInsts);

  int addTestAlpha(const vector<Instance>& vecInsts);

  void extractLinearFeatures(vector<string>& features, const Instance* pInstance, int idx);
  void extractFeature(Feature& feat, const Instance* pInstance, int idx);

  void convert2Example(const Instance* pInstance, Example& exam);
  void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

public:
  void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile);
  int predict(const vector<Feature>& features, const int& left, const int& right, vector<string>& outputs, const vector<string>& words);
  void test(const string& testFile, const string& outputFile, const string& modelFile);

  void writeModelFile(const string& outputModelFile);
  void loadModelFile(const string& inputModelFile);

};

#endif /* SRC_NNCRF_H_ */
