# GEIA
Code for reproducing Findings-ACL 2023 paper: [Sentence Embedding Leaks More Information than You Expect: Generative Embedding Inversion Attack to Recover the Whole Sentence](https://aclanthology.org/2023.findings-acl.881/)


### Package Dependencies
To install the environment needed for training the model and running the evaluations:
```
conda env create -f env.yml
```

after it has finished installing the packages, the environment can be activated using:

```
conda activate GEIA
```

### Data Preparation
We upload personachat data under the ```data/``` folder.

For other datasets, we use ```datasets``` package to download and store them, so you can run our code directly.

### GEIA
**You need to set up arguments properly before running scripts**:

```python attacker.py```

* --attack_model: attacker model (decoder) name (default value is 'microsoft/DialoGPT-medium')
* --embed_model: embedding model (sentence embedding) name (default value is 'sent_t5_large')
* --num_epochs: number of training epochs (default value is 10)
* --batch_size: batch_size (default value is 64)
* --dataset: name of the dataset including personachat, qnli, mnli, sst2, wmt16, multi_woz and abcd (default value is 'personachat')
* --data_type: train or test
* --beam: toggles beam search decoding (default value is True)

Refer to the model card in `attacker.py` to view the different model paths.

By running:
```python projection.py```
You will train your own baseline model and evaluate it. If you want to just train or eval a certain model, check the last four lines of ```projection.py``` and disable the corresponding codes.


### Evaluation
**You need to make sure the test reuslt paths is set inside the 'eval_xxx.py' files.**

To obtain classification performance, run:
```python eval_classification.py```
* --attack_model: attacker model (decoder) name (default value is 'microsoft/DialoGPT-medium')
* --embed_model: embedding model (sentence embedding) name (default value is 'sent_t5_large')
* --dataset: name of the dataset including personachat, qnli, mnli, sst2, wmt16, multi_woz and abcd (default value is 'personachat')
* --beam: toggles beam search decoding (default value is True)

To obtain generation performance, run:
```python eval_generation.py```

* --attack_model: attacker model (decoder) name (default value is 'microsoft/DialoGPT-medium')
* --embed_model: embedding model (sentence embedding) name (default value is 'sent_t5_large')
* --num_epochs: number of training epochs (default value is 10)
* --batch_size: batch_size (default value is 64)
* --dataset: name of the dataset including personachat, qnli, mnli, sst2, wmt16, multi_woz and abcd (default value is 'personachat')
* --beam: toggles beam search decoding (default value is True)


To calculate perplexity, you need to set the LM to caluate PPL, run:
```python eval_ner.py```

* --attack_model: attacker model (decoder) name (default value is 'microsoft/DialoGPT-medium')
* --embed_model: embedding model (sentence embedding) name (default value is 'sent_t5_large')
* --dataset: name of the dataset including personachat, qnli, mnli, sst2, wmt16, multi_woz and abcd (default value is 'personachat')
* --beam: toggles beam search decoding (default value is True)
