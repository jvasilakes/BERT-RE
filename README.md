# BERT for Relation Extraction

`bert_re` contains Tensorflow 2 implementations of BERT-based models
for relation extraction, as given in Figure 3 of
[Matching the Blanks: Distributional Similarity for Relation Learning](https://www.aclweb.org/anthology/P19-1279/)

These models are:

| Model  | Input           | Classification Head   |
|--------|-----------------|-----------------------|
|  3a    | Standard        |  CLS Token            |
|  3b    | Standard        |  Mention Pooling      |
|  3c    | Positional Emb. |  Mention Pooling      |
|  3d    | Entity Markers  |  CLS Token            |
|  3e    | Entity Markers  |  Mention Pooling      |
|  3f    | Entity Markers  |  Entity Start Token   |

Model 3c is not yet implemented here.


## Installation and Usage

```
git clone https://github.com/jvasilakes/BERT-RE.git
cd BERT-RE
pip install -r requirements.txt
python setup.py develop
```

See `examples.py` for usage.


## SemEval2010 Task 8

```
python run_semeval.py --model_id ${MODEL_ID} \
		      --bert_model_dir /path/to/uncased_L-12_H-768_A-12/ \
		      --train_file /path/to/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT \
		      --test_file data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT \
		      --outdir /path/to/desired/output/directory/ \
		      --learning_rate 3e-5 --batch_size 16 --epochs 10
```

Where `${MODEL_ID}` is one of `3a, 3b, 3d, 3e, 3f`.

Model `3c` is not yet implemented.

The training dataset of 8000 examples was randomly split into 80% training (6400 examples)
and 20% (1600 examples) using `sklearn.train_test_split` with `random_state=0`.

Each model was trained with the following hyper-parameters:

 * **BERT model**: BERT-base
 * **Learning rate**: Linear warmup to 3e-5 at step 400 followed by polynomial decay.
 * **Batch size**: 16
 * **Optimizer**: Adam (epsilon = 1e-8)
 * **Loss**: Categorical cross entropy from softmax activations
 * **Classification layer dropout rate**: 0.1
 * **Epochs**: Early stopping monitoring validation loss.

Note that unlike the paper, we use a dense layer with softmax activations for computing output probabilities,
rather than layer normalization with linear activations. 


### Results

We report weighted averages of precision, recall, and F1. The number of training epochs reported refers to the number
of epochs completed before validation loss stopped improving.

| Model  | Precision | Recall   |  F1    | Support | # Train epochs |
|--------|-----------|----------|--------|---------|----------------|
|  3a    |    0.73   |  0.74    | 0.73   |  2717   |       3        |
|  3b    |    0.83   |  0.82    | 0.82   |  2717   |       4        |
|  3d    |    0.81   |  0.82    | 0.81   |  2717   |       3        |
|  3e    |    0.82   |  0.82    | 0.82   |  2717   |       4        |
|  3f    |    0.82   |  0.83    | 0.82   |  2717   |       3        |



## Acknowledgments

SemEval2010 Task 8 data obtained from [sahitya0000](https://github.com/sahitya0000/Relation-Classification)

Pretrained BERT-base weights obtained from [the official Google release](https://github.com/google-research/bert)

BERT layer implemented using [BERT for Tensorflow v2](https://github.com/kpe/bert-for-tf2)

Tokenization borrowed from the [Hugging Face Transformers Library](https://github.com/huggingface/transformers/tree/master/src/transformers)
