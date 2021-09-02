# GeoGail

Synthesized human trajectories are crucial for a large number of applications. Existing solutions are mainly based on the generative adversarial network (GAN), which is limited due to the lack of modeling the human decision-making process. In this paper, we propose a novel imitation learning based method to synthesize human trajectories.  This model utilizes a novel semantics-based interaction mechanism between the decision-making strategy and visitations to diverse geographical locations to model them in the semantic domain in a uniform manner. To augment the modeling ability to the real-world human decision-making policy, we propose a feature extraction model to extract the internal latent factors of variation of different individuals, and then propose a novel self-attention based policy net to capture the long-term correlation of mobility and decision-making patterns. Then, to better reward users' mobility behavior, we propose a novel multi-scale reward net combined with mutual information to model the instant reward, long-term reward, and individual characteristics in a cohesive manner. Extensive experimental results on two real-world trajectory datasets show that our proposed model can synthesize the most high-quality trajectory data compared with six state-of-the-art baselines in terms of a number of key usability metrics, and can well support practical applications based on trajectory data, demonstrating its effectiveness. What's more, our proposed method can learn explainable knowledge automatically from data, including explainable statistical features of trajectories and statistical relation between decision-making policy and features.
![Result](Framework_Overall2.png.png?raw=true)

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
	- [Data](#data)
	- [Model](#model)
	- [Quantiazationmethod](#quantiazationmethod)
- [Contributors](#contributors)
- [License](#license)

## Background

With the exponential growth of computational power and the invention of various novel models, Deep Neural Network (DNN) has led to a series of breakthroughs in many different fields. Nowadays, DNNs could achieve groundbreaking performance in countless applications such as style transfer, classifications, haze removal and other applications. However, the deployment of these high-performance models onto resource-constrained devices (e.g., mobile devices with limited storage capacity and computation capability) hinders usage due to their high storage complexity. For instance, ResNet needs more than $10^7$ bits for weight storage to achieve an accuracy of 70\% on the ImageNet dataset.

Neural network quantization provides a promising solution to this problem since it could drastically reduce the model size while preserving most network performance. Quantization is an active area of research and development, enabling neural networks to be practically usable and more efficient in various domains. Out of all the quantization paradigms, bitwise quantization is one of the most successful and prominent approaches to quantization. Instead of designing novel network architectures that exploits memory-efficient operations, bitwise quantization targets parameters of existing models and reduce them from 32bit floating point into lower bit-depth representations. However, the performance of bitwise quantization methods depends on many attributes of the model, such as depth and width. Furthermore, these bitwise quantization methods use stable mapping functions and mapping intervals, leading to their lack of generality and potential waste of scarce bits. 

In this project, we proposed a novel quantization method based on k-means clustering. Given the network parameters, our approach will generate problem-specific k-mean clustering that minimizes the loss of the model based on the training gradient of the network. Our method will then quantize the entire model by remapping all the network parameters to their corresponding centroids. 

## Install

This project uses pytorch and jyupter notebook. Go check them out if you don't have them locally installed.


## Usage

This is only a documentation package.


### Data

To get the data for the model.
#### Data for MLP
[train|dev]_labels.npy contain a numpy object array of shape [utterances]. Each element in the array is an int32 array of shape [time] and provides the phoneme state label for each frame. There are 71 distinct labels [0-70], one for each subphoneme.

```sh
$ kaggle competitions download -c 11785-spring2021-hw1p2
```
#### Data for CNNs
Given an image of a person’s face, the task of classifying the ID of the face is known as face classification. The input to your system will be a face image and you will have to predict the ID of the face. The ground truth will be present in the training data and the network will be doing an
Multi-class Classification to get the prediction. You are provided with a validation set for fine-tuning your model.

```sh
$ kaggle competitions download -c 11785-spring2021-hw2p2s1-face-classification
```
### Model

Toy model of MLP and CNN
#### MLP structure 
![Result](baseline_model.png?raw=true)

```sh
$ run MLP.ipynb
```
#### CNN structure 
![Result](Resnet34.png?raw=true)

```sh
$ run CNN--resnet34.ipynb
```
### Quantiazationmethod

Kmeans method for quantization


### Contributors

This project exists thanks to all the people who contribute. 

## License

[MIT](LICENSE)
