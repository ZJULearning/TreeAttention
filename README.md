# TreeAttention

Table of Contents
=================
<!--ts-->
* [Introduction](#introduction)
* [Performance](#performance)
	 * [Datasets](#datasets)
	 * [Compared Algorithms](#compared-algorithms)
	 * [Results](#results)
* [Building Instruction](#building-instruction)
	 * [Prerequisites](#prerequisites)
* [Usage](#usage)
* [Reference](#reference)
* [License](#license)
<!--te-->

## Introduction
The code for [**A Better Way to Attend: Attention with Trees for Video Question Answering**](https://ieeexplore.ieee.org/document/8419716)

The HTreeMN model is a tree-structured attention neural network based on the syntactic parse tree of the natural language sentence. Each node of the tree-structured network does its computation based on the property of the corresponding word or intermediate representation.

![model](https://github.com/xuehy/TreeAttention/blob/master/overview.png)

For a faster partially batched version of the model, see [BatchedTreeLSTM](https://github.com/xuehy/BatchedTreeLSTM)

## Performance

### Datasets

+ [VideoQA Dataset](https://github.com/xuehy/videoqa)

### Compared Algorithms


+ [E-SA] (https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14906/14319)
+ [E-SS] (https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14906/14319) 
+ Simple: a designed based-line, which does not utilize attention mechanisms.


### Results
HTreeMN achieves the best results. Its performance does not drop as the length of question increases.

![table](https://github.com/ZJULearning/TreeAttention/edit/master/r.png)


## Building Instruction

### Prerequisites

+ Python 3.0+
+ Pytorch 0.4.0+


## Usage
+ Packaging the datasets into python pickle files and run ``` python main.py ```

## Reference
If you use our work, please cite our paper,
```
@article{xue2018tree,

title={A Better Way to Attend: Attention With Trees for Video Question Answering},

author={Xue, Hongyang and Chu, Wenqing and Zhao, Zhou and Cai, Deng},

journal={IEEE Transactions on Image Processing},

year={2018},

publisher={IEEE}

}
```



