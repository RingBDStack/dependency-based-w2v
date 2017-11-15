//  2017, Modified by Lin Ziwei and Li Chen on word2vec in 
//  Advanced Computer Technology(ACT), Beihang University
// Added:
//   - support for different order of dependencies for HS&CBOW & NS&SG
//   - different input context
//  See readme.md on https://github.com/BUAA-ACT-507/Dependency-Based-Word-Embedding

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_LINE 10000
#define MAX_CODE_LENGTH 40
#define MAX_YICUN 6000                 //amount of dependencies(weights)
#define PI (atan(1.0) * 4)

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
	long long cn;
	int *point;
	char *word, *code, codelen;
};

typedef struct node
{
	long long word;
	real score;
	long long dep[10];			   //all dependencies of this word
	long long jie;                     //order of dependence
	struct node *next;
}Node, *sNode;

char train_file[MAX_STRING], output_file[MAX_STRING], new_output_file[MAX_STRING], weight_output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING], read_weightcn[MAX_STRING];
real weight[MAX_YICUN];               //all weights
int weightcn[MAX_YICUN];              //frequency of weights
real multi[10] = {1, 1.2, 1.4, 1.8, 2.5, 3.4, 5, 6, 7, 8};                       //preset value
real premulti[10] = {1, 0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05};         //preset value
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1, new_operation = 0;

int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100, weight_layer_size = 50;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0, train_weights = 0, weight_size = 0;
real alpha = 0.015, starting_alpha, sample = 1e-3, weight_sample = 1e-10;
real *syn0, *syn1, *syn2, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
	int a, i;
	long long train_words_pow = 0;
	real d1, power = 0.75;
	table = (int *)malloc(table_size * sizeof(int));
	for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
	i = 0;
	d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (real)table_size > d1) {
			i++;
			d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
		}
		if (i >= vocab_size) i = vocab_size - 1;
	}
}

void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				if (ch == '\n') ungetc(ch, fin);
				break;
			}
			if (ch == '\n') {
				strcpy(word, (char *)"</s>");
				return;
			}
			else continue;
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1) a--;   // Truncate too long words
	}
	word[a] = 0;
}

int ReadNum(FILE* fin){
	int readnum = 0;
	int count = 0;
	char ch;
	while (!feof(fin)){
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')){
			if (count > 0) {
				if (ch == '\n')  break;
			}
			if (ch == '\n') { 
				return 0;
			}
			else continue;
		}
		if (isdigit(ch)){
			readnum = 10 * readnum + (ch - '0');
		}
		count++;
		if (count >= MAX_STRING - 1) count--;
	}
	return readnum;
}

int GetWordHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % vocab_hash_size;
	return hash;
}

int SearchVocab(char *word) {
	unsigned int hash = GetWordHash(word);
	if (strcmp(word, "</s>") == 0) return -2;
	while (1) {
		if (vocab_hash[hash] == -1) return -1;
		if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;
	return SearchVocab(word);
}

int ReadNewWord(Node* tail, char *line, long long b, long long a) {
	long long i, c, num, jie, status = 0;
	real score = 1;
	char word[MAX_STRING];
	memset(word, 0, MAX_STRING);
	sNode temp = (sNode)malloc(sizeof(Node));
	i = c = num = jie = status = 0;
	for (c = 0; c < 10;c++)  temp->dep[c] = -1;
	c = status = num = jie = i = 0;
	for (i = b;i < a;i++) {
		if (line[i] == ' ' || line[i] == '\t')  continue;
		if (status == 0) {
			if (isalpha(line[i])) {
				word[c] = line[i];
				c++;
				if (c >= MAX_STRING - 1)  c--;
			}
			if (!isalpha(line[i+1]) || i == a - 1) {
				status = 1;
				word[c] = 0;
				temp->word = SearchVocab(word);
			}
		}
		else if (status == 1) {
			if (line[i] == ',') {
				temp->dep[jie++] = num;
				num = 0;
			}
			if(i < a){
				if (i + 1 == a - 1 && isdigit(line[i])){
					num = num * 10 + (line[i] - '0');
					temp->dep[jie++] = num;
					num = 0;
					i++;
					break;
				}
				else if(isdigit(line[i]) && (!isdigit(line[i+1])) && line[i+1] != ',' && (line[i + 1] == ' ')) {
					num = num * 10 + (line[i] - '0');
					temp->dep[jie++] = num;
					num = 0;
					i++;
					break;
				}
				else if (isdigit(line[i]) && i + 1 != a - 1 && line[i + 1] != ' '){
					num = num * 10 + (line[i] - '0');
				}
			}
		}
	}
	score = 1;
	for (c = 0;c < 10;c++) {
		if (temp->dep[c] == -1) {
			continue;
		}
		else {
			num = temp->dep[c];
			score *= (weight[num]);
			num = c;
		}
	}
	for (status = 0;status <= num;status++)	 score /= multi[status];
	temp->score = score;
	temp->jie = jie;
	temp->next = NULL;
	tail->next = temp;
	tail = tail->next;
	return i;
}

// Reads a line and get all scores
void GetScore(FILE *fin, Node* head, Node* tail) {
	long long a, b, c, status;
	char ch;
	char word[MAX_STRING];
	char line[MAX_LINE];
	memset(line, 0, MAX_LINE);
	if (feof(fin)) return;
	a = b = c = status = 0;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13) continue;
		if (ch == '\n' || feof(fin)) {
			if (a == 0) {
				line[0] = '1';
				return;
			}
			line[a] = '\0';
			a++;
			status = 0;
			for (b = c = 0;b < a;b++) {
				if (status == 0 && isalpha(line[b])) {
					word[c] = line[b];
					c++;
					if (c >= MAX_STRING - 1) c--;
				}
				if (status == 0 && (!isalpha(line[b]))) {
					status = 1;
					word[c] = 0;
					tail->word = SearchVocab(word);
				}
				if (status == 1) {
					b = ReadNewWord(tail, line, b, a);
					if (tail->next->word == -1){
						free(tail->next);
						tail->next = NULL;
					}
					else{tail = tail->next;}
				}
			}
			if (a > 0) {
				if (ch == '\n') return;
			}
			return;
		}
		else {
			if(a < MAX_LINE - 1)
			{
				line[a] = ch;
				a++;
			}
			else{continue;}
		}
	}
	return;
}

int AddWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = 0;
	vocab_size++;
				 // Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_size - 1;
	return vocab_size - 1;
}

int VocabCompare(const void *a, const void *b) {
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

void SortVocab() {
	int a, size;
	unsigned int hash;
	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	size = vocab_size;
	train_words = 0;
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		if ((vocab[a].cn < min_count) && (a != 0)) {
			vocab_size--;
			free(vocab[a].word);
		}
		else {
			  // Hash will be re-computed, as after the sorting it is not actual
			hash = GetWordHash(vocab[a].word);
			while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
			vocab_hash[hash] = a;
			train_words += vocab[a].cn;
		}
	}
	vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
	for (a = 0; a < vocab_size; a++) {
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

void ReduceVocab() {
	int a, b = 0;
	unsigned int hash;
	for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
		vocab[b].cn = vocab[a].cn;
		vocab[b].word = vocab[a].word;
		b++;
	}
	else free(vocab[a].word);
	vocab_size = b;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++) {
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	fflush(stdout);
	min_reduce++;
}

void CreateBinaryTree() {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
	for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
	pos1 = vocab_size - 1;
	pos2 = vocab_size;
	for (a = 0; a < vocab_size - 1; a++) {
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			}
			else {
				min1i = pos2;
				pos2++;
			}
		}
		else {
			min1i = pos2;
			pos2++;
		}
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			}
			else {
				min2i = pos2;
				pos2++;
			}
		}
		else {
			min2i = pos2;
			pos2++;
		}
		count[vocab_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		binary[min2i] = 1;
	}
	for (a = 0; a < vocab_size; a++) {
		b = a;
		i = 0;
		while (1) {
			code[i] = binary[b];
			point[i] = b;
			i++;
			b = parent_node[b];
			if (b == vocab_size * 2 - 2) break;
		}
		vocab[a].codelen = i;
		vocab[a].point[0] = vocab_size - 2;
		for (b = 0; b < i; b++) {
			vocab[a].code[i - b - 1] = code[b];
			vocab[a].point[i - b] = point[b] - vocab_size;
		}
	}
	free(count);
	free(binary);
	free(parent_node);
}

void LearnVocabFromTrainFile() {
	char word[MAX_STRING];
	FILE *fin;
	long long a, i;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	vocab_size = 0;
	AddWordToVocab((char *)"</s>");
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		train_words++;
		if ((debug_mode > 1) && (train_words % 100000 == 0)) {
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		i = SearchVocab(word);
		if (i == -1) {
			a = AddWordToVocab(word);
			vocab[a].cn = 1;
		}
		else vocab[i].cn++;
		if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
	}
	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	file_size = ftell(fin);
	fclose(fin);
}

void SaveVocab() {
	long long i;
	FILE *fo = fopen(save_vocab_file, "wb");
	for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
	fclose(fo);
}

void ReadVocab() {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	FILE *fin = fopen(read_vocab_file, "rb");
	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	vocab_size = 0;
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		a = AddWordToVocab(word);
		fscanf(fin, "%lld%c", &vocab[a].cn, &c);
		i++;
	}
	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin);
	fclose(fin);
}

void ReadWeightcn() {
	int num = 0;
	int count = 0;
	FILE *fin = fopen(read_weightcn, "rb");
	if (fin == NULL) {
		printf("Weightcn file not found\n");
		exit(1);
	}
	while (1) {
		num = 0;
		num = ReadNum(fin);
		if (feof(fin)) break;
		weightcn[count] = num;
		train_weights += num;
		count++;
	}
	weight_size = count;
	printf("Weights in train file: %d\n", count);
	fclose(fin);
}


void InitNet() {
	long long a, b;
	unsigned long long next_random = 1;
	a = posix_memalign((void **)&syn2, 128, (long long)weight_size * weight_layer_size * sizeof(real));
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
	if (syn0 == NULL) { printf("Memory allocation failed\n"); exit(1); }
	if (syn2 == NULL) { printf("Memory allocation failed\n"); exit(1); }
	if (hs) {
		a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * (layer1_size + weight_layer_size) * sizeof(real));
		if (syn1 == NULL) { printf("Memory allocation failed\n"); exit(1); }
		for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
			syn1[a * (layer1_size + weight_layer_size) + b] = 0;
	}
	if (negative>0) {
		a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * (layer1_size + weight_layer_size) * sizeof(real));
		if (syn1neg == NULL) { printf("Memory allocation failed\n"); exit(1); }
		for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
			syn1neg[a * (layer1_size + weight_layer_size) + b] = 0;
	}
	for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
		next_random = next_random * (unsigned long long)25214903917 + 11;
		syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
	}
	next_random = 1;
	for (a = 0; a < weight_size; a++) for (b = 0; b < weight_layer_size; b++) {
		next_random = next_random * (unsigned long long)25214903917 + 11;
		syn2[a * weight_layer_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / weight_layer_size;
	}
	CreateBinaryTree();
}

Node* Insert_sort(Node* head) {
	Node *first, *t, *p, *q;
	if (head == NULL || head->next == NULL) return head;
	first = head->next;
	head->next = NULL;
	while (first != NULL) {
		for (t = first, q = head; ((q!= NULL) && (q->score >= t->score)); p = q, q = q->next);
		first = first->next;
		if (q == head) 
		{
			head = t;
		}
		else{
			p->next = t;
		}
		t->next = q;
	}
	return head;
}

void *TrainModelThread(void *id) {
	long long a, b, d, z, word, last_word, sentence_length = 0;
	long long learn_word_count = 0, last_learn_word_count = 0, word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
	long long l1, l2, c, target, label, local_iter = iter;
	int randomdep[5], randomorder = 0;
	unsigned long long randseed;
	unsigned long long next_random = (long long)id;
	unsigned long long next_random_s = (long long)id;
	real f, g;
	real sum = 0, average = 0;
	int count = 0, wcount = 0, sizew = 0;
	clock_t now;
	real lamda = 0, multir = 0, multiresult = 0;
	sNode head = NULL;
	sNode tail = NULL;
	real *neu1 = (real *)calloc(layer1_size + weight_layer_size, sizeof(real));
	real *neu1e = (real *)calloc(layer1_size + weight_layer_size, sizeof(real));
	real *neu1w = (real *)calloc(layer1_size + weight_layer_size, sizeof(real));//The size of weight_layer is assumed as 50
	FILE *fi = fopen(train_file, "rb");
	FILE *new_operation_fi = fopen(train_file, "rb");
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	fseek(new_operation_fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	randseed = (unsigned long long)time(NULL) * 10 + id;
	randseed = randseed * 1103515245 + 12345; 
	while (1) {
		head = (sNode)malloc(sizeof(Node));
		tail = head;
		tail->next = NULL;
		if (learn_word_count - last_learn_word_count > 10000) {
			word_count_actual += learn_word_count - last_learn_word_count;
			last_learn_word_count = learn_word_count;
			if ((debug_mode > 1)) {
				now = clock();
				printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
					word_count_actual / (real)(iter * train_words + 1) * 100,
					word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}
			alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
			if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
		}
		if (sentence_length == 0) {
			if (new_operation == 1 && !feof(new_operation_fi)) {
				GetScore(new_operation_fi, head, tail);//get scores of all dependencies
				head->score = 0;
				head->next = Insert_sort(head->next);
			}
			while (1) {
				word = ReadWordIndex(fi);
				if (feof(fi)) break;
				if (word == -2) break;
				if (word == -1) continue;
				word_count++;
				if (word == 0) break;
				sen[sentence_length] = word;
				sentence_length++;
				if (sentence_length >= MAX_SENTENCE_LENGTH) break;
			}
	        learn_word_count++;
		}
		if (feof(fi) || (learn_word_count > train_words / num_threads)) {
			word_count_actual += learn_word_count - last_learn_word_count;;
			local_iter--;
			if (local_iter == 0) break;
			word_count = 0;
	        last_learn_word_count = 0;
	        learn_word_count = 0;
			sentence_length = 0;
			fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
			fseek(new_operation_fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
			continue;
		}
		word = sen[0];
		if (word == -1) continue;
		for (c = 0; c < layer1_size + weight_layer_size; c++) neu1[c] = 0;
		for (c = 0; c < layer1_size + weight_layer_size; c++) neu1e[c] = 0;
		for (c = 0; c < layer1_size + weight_layer_size; c++) neu1w[c] = 0;
		next_random_s = next_random_s* (unsigned long long)25214903917 + 11;
		b = next_random_s % window + 1;
		if (cbow) {
			sum = 0;
			count = 0;
			if (new_operation == 1) {
				Node *n,*p;
				Node *h;
				h = head;
				for(;h->next!=NULL;)//negative sampling of dependencies
				{
					if (sample > 0) {
						real ran = (sqrt(vocab[h->next->word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[h->next->word].cn;
						next_random_s = next_random_s * (unsigned long long)25214903917 + 11;
						if (ran < (next_random_s & 0xFFFF) / (real)65536) {
								p = h->next;
								h->next = h->next->next;
								free(p);
						}
						else{
							h = h->next;
						}
					}
				}
				h = head;
				n = head->next;
				average = 0;
				sum = 0;
				count = 0;
				for (a = 1; a < window * 2 - b * 2 && n!=NULL ; a++,n=n->next) if (a != 0) {
					if(c < 0) continue;
					last_word = n->word;
					if (last_word == -1) {
						continue;
					}
					for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size] * n -> score;
					for (c = 0;c < 10;c++) {
						if (n->dep[c] == -1) {
							break;
						}
						else {
							for (d = layer1_size; d < weight_layer_size + layer1_size; d++){
								neu1[d] += syn2[(d - layer1_size) + (n->dep[c] * weight_layer_size)] * premulti[c];
							}
							wcount++;
						}
					}
					if (c != 0) {wcount = c;} else{wcount = 1;}	
					for (c = layer1_size; c < weight_layer_size + layer1_size; c++) neu1[c] /= wcount;
					count++;
					sum += n -> score;
				}
				if (count != 0)	average = sum / count;
				if (average == 0) average = 1;
			}
			if (count) {
				for (c = 0; c < layer1_size; c++) neu1[c] /= count;
				if (hs) for (d = 0; d < vocab[word].codelen; d++) {
					f = 0;
					l2 = vocab[word].point[d] * layer1_size;
					// Propagate hidden -> output
					for (c = 0; c < layer1_size + weight_layer_size; c++) f += neu1[c] * syn1[c + l2];
					if (f <= -MAX_EXP) continue;
					else if (f >= MAX_EXP) continue;
					else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
					// 'g' is the gradient multiplied by the learning rate
					g = (1 - vocab[word].code[d] - f) * alpha;
					// Propagate errors output -> hidden
					for (c = 0; c < layer1_size + weight_layer_size; c++){
						neu1e[c] += g * syn1[c + l2] * average;
						neu1w[c] += g * syn1[c + l2];
						// Learn weights hidden -> output
						syn1[c + l2] += g * neu1[c];
					}
				}
				if (negative > 0) for (d = 0; d < negative + 1; d++) {
					if (d == 0) {
						target = word;
						label = 1;
					}
					else {
						next_random_s = next_random_s * (unsigned long long)25214903917 + 11;
						target = table[(next_random_s >> 16) % table_size];
						if (target == 0) target = next_random_s % (vocab_size - 1) + 1;
						if (target == word) continue;
						label = 0;
					}
					l2 = target * layer1_size;
					f = 0;
					for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
					if (f > MAX_EXP) g = (label - 1) * alpha;
					else if (f < -MAX_EXP) g = (label - 0) * alpha;
					else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
					for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2] * average;
					for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];																 
				}
				if (new_operation == 1){
					last_word = sen[0];
					Node *n;
					n = head->next;
					sizew = count;
					while(sizew > 0 && n!= NULL){
						lamda = 1;
						multir = 0;
						multiresult = 0;
						for (c = 0; c < layer1_size; c++) multir += (neu1w[c] * syn0[c + n->word * layer1_size]);
						for (c = 0;c < 10;c++) {
							if (n->dep[c] == -1) {
								break;
							}
							else {
								for (d = layer1_size; d < weight_layer_size + layer1_size; d++){
									multir += (neu1w[c] * syn2[(d - layer1_size) + (n->dep[c] * weight_layer_size)]);
								}
							}
						}
						for(a = 0;a < 10;a++){
							if (n->dep[a] == -1) break;
							lamda *= multi[a];
							multiresult = multir/count;
							if (weight_sample > 0) {
								real ran = (sqrt(weightcn[n->dep[a]] / (weight_sample * train_weights)) + 1) * (weight_sample * train_weights) / weightcn[n->dep[a]];
								next_random = next_random * (unsigned long long)25214903917 + 11;
								if (ran < (next_random & 0xFFFF) / (real)65536){ 
									continue;
								}else{
									weight[n->dep[a]] += (multiresult / lamda); 
								}
							}
						}
						sizew--;
						n = n->next;
					}
				}
				Node *nn;
				nn = head->next;
				for (a = 1; a <= b && nn!=NULL; a++,nn=nn->next){
					c = a;
					if (c < 0) continue;
					if (c >= sentence_length) continue;
					last_word = nn->word;
					if (last_word == -1) continue;
					for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
					for (c = 0;c < 10;c++) {
						if (nn->dep[c] == -1) {
							break;
						}
						else {
							for (d = layer1_size; d < weight_layer_size + layer1_size; d++){
								if (weight_sample > 0) {
								real ran = (sqrt(weightcn[nn->dep[c]] / (weight_sample * train_weights)) + 1) * (weight_sample * train_weights) / weightcn[nn->dep[c]];
								next_random = next_random * (unsigned long long)25214903917 + 11;
								if (ran < (next_random & 0xFFFF) / (real)65536){ 
										continue;
									}else{
										syn2[(d - layer1_size) + nn->dep[c] * weight_layer_size] += neu1e[d];
									}
								}
							}
						}
					}
				}
			}
		}
		else {
		if (new_operation == 1) {
				Node *n,*p;
				Node *h;
				h = head;
				next_random_s = next_random_s * (unsigned long long)25214903917 + 11;
				for(;h->next!=NULL;)
				{
					if (sample > 0) {
						real ran = (sqrt(vocab[h->next->word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[h->next->word].cn;
						next_random_s = next_random_s * (unsigned long long)25214903917 + 11;
						if (ran < (next_random_s & 0xFFFF) / (real)65536) {
								p = h->next;
								h->next = h->next->next;
								free(p);
						}
						else{
							h = h->next;
						}
					}
				}
			n = head;			
			for (a = 1; a < window * 2 - b * 2 && n!=NULL; a++,n=n->next) if (a != 0) {
				c = a;
				if(c < 0) continue;
				if (c >= sentence_length) continue;
				last_word = n->word;
				if (last_word == -1) continue;
				l1 = last_word * layer1_size;
				for (c = 0; c < layer1_size + weight_layer_size; c++) neu1e[c] = 0;
				// HIERARCHICAL SOFTMAX
				if (hs) for (d = 0; d < vocab[word].codelen; d++) {
					f = 0;
					l2 = vocab[word].point[d] * layer1_size;
					// Propagate hidden -> output
					for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
					if (f <= -MAX_EXP) continue;
					else if (f >= MAX_EXP) continue;
					else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
					// 'g' is the gradient multiplied by the learning rate
					g = (1 - vocab[word].code[d] - f) * alpha;
					// Propagate errors output -> 
					for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
					// Learn weights hidden -> output
					for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
				}
				// NEGATIVE SAMPLING
				if (negative > 0) for (d = 0; d < negative + 1; d++) {
					if (d == 0) {
						target = word;
						label = 1;
					}
					else {
						next_random_s = next_random_s * (unsigned long long)25214903917 + 11;
						target = table[(next_random_s >> 16) % table_size];
						if (target == 0) target = next_random_s % (vocab_size - 1) + 1;
						if (target == word) continue;
						label = 0;
						randseed = randseed * 1103515245 + 12345;
						randomorder = ((randseed << 16) | ((randseed >> 16) & 0xFFFF))%(4 - 1 + 1) + 1;//order of negative sample
						for (c = 0;c < randomorder;c++){
							while (1){
								randseed = randseed * 1103515245 + 12345;
								randomdep[c] = ((randseed << 16) | ((randseed >> 16) & 0xFFFF))%(5999 - 0 + 1) + 0;//dependencies of negative sample
								if (weightcn[randomdep[c]] == 0){
									continue;
								}else{
									break;
								}
							}
						}
					}
					l2 = target * (layer1_size);
					f = 0;
					for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
					if (d == 0){
						for (c = 0;c < 10;c++) {
							if (n->dep[c] == -1) {
								break;
							}
							else {
								for (z = layer1_size; z < weight_layer_size + layer1_size; z++) f += syn2[z - layer1_size + n->dep[c] * weight_layer_size] * syn1neg[z + l2];
							}
						}
					}else{
						for (c = 0;c < randomorder;c++){
							for (z = layer1_size; z < layer1_size + weight_layer_size; z++) f += syn2[z - layer1_size + randomdep[c] * weight_layer_size] * syn1neg[z + l2];
						}
					}
					if (f > MAX_EXP) g = (label - 1) * alpha;
					else if (f < -MAX_EXP) g = (label - 0) * alpha;
					else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
					for (c = 0; c < layer1_size + weight_layer_size; c++) neu1e[c] += g * syn1neg[c + l2];
					for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
					if (d == 0){
						for (c = 0;c < 10;c++) {
							if (n->dep[c] == -1) {
								break;
							}
							else {
								for (z = layer1_size; z < weight_layer_size + layer1_size; z++) syn1neg[z + l2] += g * syn2[z - layer1_size + (n->dep[c] * weight_layer_size)];
							}
						}
					}else{
						for (c = 0;c < randomorder;c++){
							for (z = layer1_size; z < layer1_size + weight_layer_size; z++) syn1neg[z + l2] += g * syn2[z - layer1_size + (randomdep[c] * weight_layer_size)];
						}
					}
				}
				// Learn weights input -> hidden
				for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
				for (c = 0;c < 10;c++) {
						if (n->dep[c] == -1) {
							break;
						}
						else {
							for (z = layer1_size; z < weight_layer_size + layer1_size; z++){
								if (weight_sample > 0) {
								real ran = (sqrt(weightcn[n->dep[c]] / (weight_sample * train_weights)) + 1) * (weight_sample * train_weights) / weightcn[n->dep[c]];
								next_random_s = next_random_s * (unsigned long long)25214903917 + 11;
								if (ran < (next_random_s & 0xFFFF) / (real)65536){ 
										continue;
									}else{
										syn2[(z - layer1_size) + n->dep[c] * weight_layer_size] += neu1e[z];
									}
								}
							}
						}
				}
			}
		}
		}
		Node *p = head;
		Node *q = NULL;
		for(;p!=NULL;)
		{
			q = p;
			p = p->next;
			free(q);
		}
		sentence_length = 0;
		continue;
	}
	fclose(fi);
	fclose(new_operation_fi);
	free(neu1);
	free(neu1e);
	free(neu1w);
	pthread_exit(NULL);
}

void TrainModel() {
	long a, b, c, d;
	FILE *fo;
	FILE *new_fo;
	FILE *weight_fo;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", train_file);
	starting_alpha = alpha;
	if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
	if (save_vocab_file[0] != 0) SaveVocab();
	if (read_weightcn[0] != 0) ReadWeightcn();
	if (output_file[0] == 0) return;
	if (new_output_file[0] == 0) return;
	InitNet();
	if (negative > 0) InitUnigramTable();
	start = clock();
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	fo = fopen(output_file, "wb");
	new_fo = fopen(new_output_file, "wb");
	weight_fo = fopen(weight_output_file, "wb");
	if (classes == 0) {
		fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
		fprintf(new_fo, "%lld %lld\n", weight_size, weight_layer_size);
		for (a = 0; a < vocab_size; a++) {
			fprintf(fo, "%s ", vocab[a].word);
			if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
			else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
			fprintf(fo, "\n");
		}
		for (a = 0; a < weight_size; a++) {
			fprintf(new_fo, "%ld ", a);
			for (b = 0; b < weight_layer_size; b++) fprintf(new_fo, "%lf ", syn2[a * weight_layer_size + b]);
			fprintf(new_fo, "\n");
		}
	}
	else {
		// Run K-means on the word vectors
		int clcn = classes, iter = 10, closeid;
		int *centcn = (int *)malloc(classes * sizeof(int));
		int *cl = (int *)calloc(vocab_size, sizeof(int));
		real closev, x;
		real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
		for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
		for (a = 0; a < iter; a++) {
			for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
			for (b = 0; b < clcn; b++) centcn[b] = 1;
			for (c = 0; c < vocab_size; c++) {
				for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
				centcn[cl[c]]++;
			}
			for (b = 0; b < clcn; b++) {
				closev = 0;
				for (c = 0; c < layer1_size; c++) {
					cent[layer1_size * b + c] /= centcn[b];
					closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
				}
				closev = sqrt(closev);
				for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
			}
			for (c = 0; c < vocab_size; c++) {
				closev = -10;
				closeid = 0;
				for (d = 0; d < clcn; d++) {
					x = 0;
					for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
					if (x > closev) {
						closev = x;
						closeid = d;
					}
				}
				cl[c] = closeid;
			}
		}
		// Save the K-means classes
		for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
		free(centcn);
		free(cent);
		free(cl);
	}
	for (a = 0;a < MAX_YICUN;a++){
		fprintf(weight_fo, "%f\n", weight[a]);
	}
	fclose(fo);
	fclose(new_fo);
	fclose(weight_fo);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
		printf("\t-new-output <file>\n");
		printf("\t\tUse <file> to save the new resulting word vectors / word clusters\n");
		printf("\t-weight-output <file>\n");
		printf("\t\tUse <file> to save the resulting weigths(for cbow)\n");
		printf("\t-size <int>\n");
		printf("\t\tSet size of word vectors; default is 100\n");
		printf("\t-window <int>\n");
		printf("\t\tSet max skip length between words; default is 5\n");
		printf("\t-sample <float>\n");
		printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
		printf("\t-weight-sample <float>\n");
		printf("\t\tSet threshold for occurrence of wweights. Those that appear with higher frequency in the training data\n");
		printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
		printf("\t-hs <int>\n");
		printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 12)\n");
		printf("\t-iter <int>\n");
		printf("\t\tRun more training iterations (default 5)\n");
		printf("\t-min-count <int>\n");
		printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
		printf("\t-classes <int>\n");
		printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
		printf("\t-debug <int>\n");
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		printf("\t-save-vocab <file>\n");
		printf("\t\tThe vocabulary will be saved to <file>\n");
		printf("\t-read-vocab <file>\n");
		printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		printf("\t-read-weightcn <file>\n");
		printf("\t\tThe weight's cn will be read from <file>, not constructed from the training data\n");
		printf("\t-cbow <int>\n");
		printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
		printf("\t-new_operation <int>\n");
		printf("\t\tUse new_operation to train words model; default is 0 (use 1 for using)\n");
		printf("\nExamples:\n");
		printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
		return 0;
	}
	output_file[0] = 0;
	new_output_file[0] = 0;
	weight_output_file[0] = 0;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;
	read_weightcn[0] = 0;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-new-output", argc, argv)) > 0) strcpy(new_output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-weight-output", argc, argv)) > 0) strcpy(weight_output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-weight-sample", argc, argv)) > 0) weight_sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-new_operation", argc, argv)) > 0) new_operation = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-read-weightcn", argc, argv)) > 0) strcpy(read_weightcn, argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	if (new_operation == 1) {
		printf("new_operation\n");
		for (i = 0;i < MAX_YICUN;i++) {
			weight[i] = 0.8;
		}
	}
	for (i = 0; i <= EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}
	TrainModel();
	return 0;
}