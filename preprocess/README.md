# Preprocess

### Step 1
We should prepare a training corpus like http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2</br>
Then, run `python process_wiki.py` to filter data.

### Step 2
Using [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml) to analyze grammatical structure of sentences by running the `process.java` script. </br>
**Pay attention to set args[0] of the script as the preprocessed file in Step 1.** </br>
`parse_demo.txt` is a demo of result.</br>

### Step 3
In this step, we'll transform the output of Step 2 into trainable format.</br>
In order to do that, we've pre-stated all labeled grammatical relations in `depsl.txt`. These grammatical relations are normally fixed, you can add up them by yourself or just use the txt we provided. Then, transform these relations into their line number:</br>
All we need is to run `mdeps.java`. **Pay attention to set args[0] of the script as the preprocessed file in Step 2.**

### Step 4
The reason why we count frequency of grammatical relations is for negative sampling during training.</br>
Run `python count.py --file [file] --quantity [quantity]` to count the frequency of every relation and get the `weightcn.txt`.</br>
Run `python vocab.py --file [file]` to count the frequency of every vocabulary and get the `vocab.txt`.</br>
**[file] means the preprocessed file in Step 3(such as `mdeps-demo.txt`), [quantity] means number of lines of `depsl.txt`(default is 5785)**

### Conclusion
Here, we get the trainfile in Step3, the `weightcn.txt` and `vocab.txt` in step4. That's all we need for training.
