make clean
make
time ./word2vec -train deps3_ww.txt -output vectors_deps.bin -new-output new_vectors_deps.bin -weight-output weights_result.txt -read-vocab vocab.txt -read-weightcn weightcn.txt -cbow 0 -size 300 -window 5 -negative 4 -hs 0 -sample 1e-4 -weight-sample 1e-10 -threads 500 -binary 0 -iter 10 -new_operation 1
