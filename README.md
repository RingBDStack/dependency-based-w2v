# Dependency-Based-Word-Embedding

### Introduction
This repository contains the code for generating dependency-based word embedding modified from

* [Word2vec](https://code.google.com/archive/p/word2vec/)

Some training data can be found here: http://mattmahoney.net/dc/enwik9.zip http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

### Requirements
* Linux system
* [English Wikipedia Database](https://link.zhihu.com/?target=http%3A//download.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2) : large-scale corpus

### Setting up
* Data preprocess
  * Tool: [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml)
  * An Example for pre-trained corpus: `institutions voluntary 4 based 4652 societies 4652,5782 advocates 4652,5782,3350`
* Count words and dependencies
  * Get all word dependencies as `depsl.txt`
  * Get the result of words counting as `vocab.txt`
  * Get the result of dependencies counting as `weightcn.txt`
  * You should prepare data as `demo-depsl.txt`、`demo-vocab.txt`、`demo-weightcn.txt`
* More specific steps are in the preprocess folder
  
## Training Instructions
* Experiment configurations are found in `demo-word.sh`
* Choose an experiment that you would like to run (modify demo-word.sh, run as `./demo-word.sh`)
  * For CBOW based on HS, run `time ./word2vec -train <trainfile> -output <outputfile> -new-output <extra-dimension-outputfile> -weight-output <weight(dependencites)-outputfile> -read-vocab vocab.txt -read-weightcn weightcn.txt -cbow 1 -size 300 -window 5 -negative -1 -hs 1 -sample 1e-4 -weight-sample 1e-10 -threads 100 -binary 0 -iter 10 -new_operation 1`
  * For SG based on NS, run `time ./word2vec -train <trainfile> -output <outputfile> -new-output <extra-dimension-outputfile> -weight-output <weight(dependencites)-outputfile> -read-vocab vocab.txt -read-weightcn weightcn.txt -cbow 0 -size 300 -window 5 -negative 4 -hs 0 -sample 1e-4 -weight-sample 1e-10 -threads 200 -binary 0 -iter 10 -new_operation 1`
* Parameters
  * -new-output:50 dimension weight based embeddings 
  * -weight-output:weights of all dependencies
  * -read-weightcn:input-frenquency of all dependencies
  * -new_operation:means our modifies
 
## Our Contributes 
  * Add dependencies in word embedding training
  * New 50 demension weight based embeddings
  
## Other Tips 
* Each line in weightcn.txt is corresponding to each line in `depsl.txt` (Just in this corpus. You should get your own depsl.txt and weightcn.txt)
* The number of threads in parameter `-threads` is according to the performance of your computer, we recommand 100~500.
