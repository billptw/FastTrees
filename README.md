# FastTrees

This repository contains the code adapted from for word-level language model and unsupervised parsing experiments in 
[Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks](https://arxiv.org/abs/1810.09536) paper, 
originally forked from the 
[LSTM and QRNN Language Model Toolkit for PyTorch](https://github.com/salesforce/awd-lstm-lm).


## Software Requirements
Python 3.6, NLTK and PyTorch 0.4 are required for the current codebase.

## Steps

1. Install PyTorch 0.4 and NLTK

2. Download PTB data. Note that the two tasks, i.e., language modeling and unsupervised parsing share the same model 
structure but require different formats of the PTB data. For language modeling we need the standard 10,000 word 
[Penn Treebank corpus](https://github.com/pytorch/examples/tree/75e435f98ab7aaa7f82632d4e633e8e03070e8ac/word_language_model/data/penn) data, 
and for parsing we need [Penn Treebank Parsed](https://catalog.ldc.upenn.edu/LDC99T42) data. Place train, test and valid.txt in ./data/penn/

3. Scripts and commands

  	+  Train Language Modeling with FastTrees
  	```python main.py --model FT --batch_size 20 --dropout 0.45 --dropouth 0.3 --dropouti 0.5 --wdrop 0.45 --chunk_size 10 --seed 141 --epoch 1000 --data /path/to/your/data```

    +  Train Language Modeling with Conv FastTrees
  	```python main.py --model FFT --batch_size 20 --dropout 0.45 --dropouth 0.3 --dropouti 0.5 --wdrop 0.45 --chunk_size 10 --seed 141 --epoch 1000 --data /path/to/your/data```

    +  Train Language Modeling with Faster FastTrees
  	```python main.py --model CFT --batch_size 20 --dropout 0.45 --dropouth 0.3 --dropouti 0.5 --wdrop 0.45 --chunk_size 10 --seed 141 --epoch 1000 --data /path/to/your/data```
  
    +  Train Language Modeling with ON-LSTM
  	```python main.py --model ON --batch_size 20 --dropout 0.45 --dropouth 0.3 --dropouti 0.5 --wdrop 0.45 --chunk_size 10 --seed 141 --epoch 1000 --data /path/to/your/data```

    +  Train Language Modeling with LSTM
  	```python main.py --model LSTM --batch_size 20 --dropout 0.45 --dropouth 0.3 --dropouti 0.5 --wdrop 0.45 --chunk_size 10 --seed 141 --epoch 1000 --data /path/to/your/data```
    
  	+ Test Unsupervised Parsing
    ```python test_phrase_grammar.py --cuda```
    
