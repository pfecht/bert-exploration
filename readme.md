# BERT exploration

Workspace to explore classification and other tasks based on the [pytorch implementation](https://github.com/huggingface/pytorch-pretrained-BERT) of the original bert paper (https://arxiv.org/abs/1810.04805)


# GLUE 
See [this notebook](https://colab.research.google.com/drive/1Qc0JOJ3x4vUU3nNTtBoD5GONrrrOtNdr) for an implementation of the GLUE tasks.


# SWAG
The Situations With Adversarial Generations (SWAG) dataset contains 113k sentence-pair com- pletion examples that evaluate grounded common- sense inference (Zellers et al., 2018). Given a sentence from a video captioning
dataset, the task is to decide among four choices the most plausible continuation. 

Running

```bash 
export SWAG_DIR=/home/pfecht/thesis/swagaf

python run_swag.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_eval \
  --data_dir $SWAG_DIR/data \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 80 \
  --output_dir /home/pfecht/tmp/swag_output/  \
  --gradient_accumulation_steps 4
```

results in


* **Accuracy**: **78.58** (BERT paper **81.6**)

```
12/14/2018 18:42:18 - INFO - __main__ -     eval_accuracy = 0.7858642407277817
12/14/2018 18:42:18 - INFO - __main__ -     eval_loss = 0.6655298910721517
12/14/2018 18:42:18 - INFO - __main__ -     global_step = 13788
12/14/2018 18:42:18 - INFO - __main__ -     loss = 0.07108418613090857
```

with fine-tuning time on a single GPU (GeForce GTX TITAN X): **around 4 hours**.


# SQuAD
> see https://github.com/huggingface/pytorch-pretrained-BERT#fine-tuning-with-bert-running-the-examples


Running

```bash
$ python run_squad.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUT_DIR \
  --optimize_on_cpu
```

* `optimize_on_CPU` is important to obtain enough space on the GPU for training. BertOptimiezer stores 2 moving averages of the weights of the model wich means We have to store 3-times the size of the model in the GPU if we don't move it to CPU.
* OOM errors are proportional to `train_batch_size` and `max_seq_length`.

results in 

* **F1 score**: 88.28 (BERT paper 88.5)
* **EM (Exact match)**: 81.05 (BERT paper = 80.8)

with fine-tuning time on a single GPU (GeForce GTX TITAN X): **around 8 hours**.

running evaluation based on

```json
$ python evaluate-v1.1.py /home/pfecht/res/SQUAD/dev-v1.1.json predictions.json
{"f1": 88.28409344840951, "exact_match": 81.05014191106906}
```