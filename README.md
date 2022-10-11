# RLRep


RLRep is a project to automatically generate program repair recommendation in the field of smart contracts for given code snippets with their contexts.
The source code and dataset are opened.

## Introduction

`multistep_RLRep.py`: the model framework for reinforcement learning and its implementation

`src/`, `config/`, `utils/` and `smartBugs.py`: integrated the *smartBugs* tool.

`similarity_compute.py` and `FastText/`: by *FastText* library, compute the similarity between the generated contract and the buggy contract. (one of the modules that make up the reward function)

`entropy_compute.py` and `entropy_compute/`: by *the concept of entropy proposed by [Ray et al.](https://ieeexplore.ieee.org/abstract/document/7886923)*, compute the entropy of the generated contract. (one of the modules that make up the reward function as well)

`utils2.py`: some useful methods of reward function and reinforcement learning.

`top300_identifier_dict.txt`: top-300 most frequent tokens in the source code.

`solidityparser/`: the Solidity lexer and parser built on top of ANTLR.

`code2ast.js` and `node_modules/`: convert source code to a preorder traversal sequence of AST.

`dataset_vul/`: dataset directory

`requirements.txt`: a file listing all the dependencies for RLRepair

## Usage

1. Install packages needed using pip:

```
pip install -r requirements.txt
```

2. make sure that all input files are ready: (you can refer to the format of our input files in `dataset_vul/newALLBUGS/`)

- mapping (map source token to index): `code_w2i.pkl`, `code_i2w.pkl`, `ast_w2i.pkl` and `ast_i2w.pkl`
- first input: `threelines-tokenseq/`
- second input: `ast/`
- data for pretraining: `pretrain/` and `pretrain_label/`
- data for validation: `validation/`

3. training and validation

```python
# for example:
python main.py dataset_vul/newALLBUGS
```

4. reult

At last, result can be got in `dataset_vul/newALLBUGS/validation/result/`.

