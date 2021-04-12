# SemiSupervisedLearning_Reproduction
This repository contains all files related to the student effort to recreate the paper 'Semi-supervised Learning with Deep Generative Models' by D.P. Kingma

## Introduction 
The aim of this blog post is to discuss the results of our experiments regarding the paper *Semi-Supervised Learning with Deep Generative Models* by D.P. Kingma, D.J. Rezende, S. Mohamed and M. Welling (2014). The experiments we have done are an extension of those in the paper. In this post we will analyze the influence that the size of the dataset has on the performance of the semi-supervised learning algorithm.

## What is semi-supervised learning?
The fuel of every machine learning model is data. Enormous amounts of data are needed to adjust what are often hundreds of parameters in such a way that a model is able to make a useful production. In order for the model to learn from all this information, a big part of machine learning relies on labelled data. This way, the prediction made by the model can be compared to the actual piece of information and thereby be considered correct or incorrect. The process of labelling data is an extensive (and often expensive) job. Luckily, there have been plenty of projects targeted at creating large labelled datasets. Perhaps the most famous example of this is MNIST; a dataset of 70,000 labelled handwritten digits, and also the dataset used for our experiments. But what if only a small subset of the data has labels available? This problem is being researched in the field of semi-supervised learning. By using a small amount of labelled data, big increases in performance can be obtained. Kingma et al. asked themselves the following question for their paper: “How can properties of the data be used to improve decision boundaries and to allow for classification that is more accurate than that based on classifiers constructed using the labelled data alone?” How they tried to answer this question, and what we did to broaden the scope of these answers will be discussed next.

## Model Construction
There are multiple approaches towards semi-supervised learning. For this blog post, we won’t go into details of all the different types, but some examples are algorithms based on a self-training scheme (Rosenberg et al., 2005), Transductive SVMs (Joachims, 1999), graph-based methods, neural network-based approaches and the Manifold Tangent Classifier (Rifai et al, 2011). All of these approaches have their own set of advantages and costs. Kingma et al. decided to use generative models for their paper. This means that the semi-supervised learning problem will try to predict the absent information from the dataset. As there are many different algorithms for semi-supervised learning, there are also a lot of approaches towards generative models to choose from. The paper argues that from these options a generalised and scalable probabilistic approach is missing. They wrote down the following four key points in which they tried to address this problem:
* Describe a new framework for semi-supervised learning with generative models, consisting of the fusion of probabilistic modelling and deep neural networks.
*
The algorithm consists of two different models. The first is the *latent-feature discriminative model (**M1**)*. This model was used to create features of the data that can be used to cluster related observations without most of the labels. M1 is itself a deep generative model, so that the learned features will be more robust than with a linear embedding. The second model is the actual *generative semi-supervised model (**M2**)*. It is a probabilistic model that uses a latent variable y combined with an independent continuous latent variable z to describe the process of generating the data. Because these variables are independent, the class specification can be separated from the writing style of the digit. The unlabelled data is classified as inference, which means that predictions about the data are being made by the existing model. When these two main models are combined, the result is a *stacked generative semi-supervised model (**M1+M2**)*. The first model is used to learn a new latent representation z1, after which the second model is learned using the embeddings of z1.

## Dependencies Between the Random Variables
In order to create probability distributions for your random variables obtained by experiment or survey, it is necessary to have independent and identically distributed random variables. The models described above do not meet this requirement, as there exist nonlinear, non-conjugate dependencies between the variables. To solve this problem, a technique was used called variational inference (Kingma and Welling, 2014; Rezende et al., 2014). This means that ...


## Reproduction & Parameter Tuning
For the reproduction assignment of the paper *Semi-supervised Learning with Deep Generative Models*, the authors of the reproduction were tasked with evaluating table 1 from the report. This table displays benchmark results of the semi-supervised classification on the MNIST dataset with a varying number of labelled training points (N).

![Screenshot 2021-04-11 at 22 20 39](https://user-images.githubusercontent.com/61149611/114320002-278aef80-9b14-11eb-8938-9ce264140f86.png)

The code, provided by the authors of the paper to reproduce some key results (https://github.com/dpkingma/nips14-ssl ), is able to reproduce only the M1+M2 stacked model for the partially observed labels. Reproducing these results is therefore what will be focussed on in this report.

As the code originates from 2014, it relies on relatively old and outdated software. For example, it relies on `python2` and the `Theano` library to perform the algorithm training process. Furthermore, the code as provided by the authors, is not GPU optimized (or cuda compatible). All this contributes to the fact that a single evaluation of the semi-supervised classification takes a very long time to run. It was observed that the M1+M2 stacked model for *N = 3000* could take as long as 48 hours to complete. 

Of course, the first solution that comes to mind is to update the code to a more modern version of python and neural network library such as Pytorch. In doing so, cuda compatibility could be added to significantly reduce the required computational effort. However, D. P. Kingma’s et. al. code is fairly elaborate, having even built their own library for efficient gradient-based inference and learning in Bayes nets, built on top of Theano (`Anglepy`). Updating the code was therefore considered to be infeasible within the provided time-frame. However, this left the authors of the reproduction wondering if, without much alteration to the code, similar levels of accuracy could be achieved using less computational effort (CPU time).

### Experiments
To do so, the authors thought of two experiments that could be performed. First of all, the default MNIST dataset used consists of 50,000 training points. The dataset is then divided into a training-, validation- and test-set to train and evaluate the deep generative model (as can be seen in the figure - retrieved from https://en.m.wikipedia.org/wiki/Training,_validation,_and_test_sets_).

![Screenshot 2021-04-11 at 22 25 04](https://user-images.githubusercontent.com/61149611/114320139-c879aa80-9b14-11eb-819b-a8317ffba0bb.png)

Of a fraction of the training set (N), the labels are used and for the remainder, the labels are discarded. The first experiment is therefore to reduce the number of training points in the input dataset while keeping the fraction of labelled data the same. This should decrease training time but analysis is needed to see if similar accuracy levels as Table 1 can be achieved.

The second experiment centres around the number of iterations (epochs) of the algorithm before it is truncated. By default, the algorithm runs to 3000 epochs without any truncation criterion. In the second experiment, it is analysed if the addition of a truncation threshold in combination with the reduced datasets could potentially even further decrease the CPU time required. As D.P. Kingma indicates himself, sufficiently accurate results can be obtained without running through all epochs. This requires further analysis!

The questions that are aimed to be answered by the two experiments can therefore be summarized as follows:

* __Can the reduction of the dataset size provide similarly accurate results at a lower computational expense__

* __Can a truncation criterium in combination with different dataset sizes ensure similarly accurate results at a lower computational expense?__

### A Priori Expectations
It is expected that the decrease of the number of training points in the dataset will significantly decrease the CPU time of a single algorithm evaluation. The reduction in number of training points that need to be evaluated will reduce the CPU time needed to run the algorithm per epoch, therefore reducing the total CPU time needed. However reducing can potentially have two effects on the results. First of all, it could well be that the error rate obtained by a reduced dataset size cannot achieve similar levels of accuracy as Table 1. Furthermore, it is expected that the decrease in the training points will lead to a decrease in the convergence rate. This will mean that it will take more epochs for the algorithm to converge. Together with the truncation threshold investigation it could well be that the algorithm evaluations for larger datasets converge at a lower epoch than the evaluations using smaller datasets. It will be interesting to see what settings will lead to the lowest total CPU time. However, it is important to note that a similar level of convergence does __not__ mean a similar level of accuracy. It could well be that smaller training datasets do converge to a similar level as the larger ones, but are unable to reach similar accuracy levels.

### Experiment Execution
To perform the experiment, a slight alteration was made to the original code. In the run file (`run_2layer_sll.py`) and subsequently, the actual execution file (`learn_yz_x_ss.py`) a function was added to specify the size of the input training set. The altered files can be found in this GitHub repository (https://github.com/frederikcde/SemiSupervisedLearning_Reproduction/tree/main/Altered_ModelFiles).

This function can be called from the terminal just as the original code was meant to be. To run the semi-supervised learning experiments with model M1+M2 use
```
python run_2layer_ssl.py [n_labels] [seed] [n_training]
```
where where `n_labels` is the number of labels, `seed` is the random seed for Numpy and `n_training` the size of the trainingset. 
#### Note
For all runs of the M1+M2 stacked model, `seed = 0` is used. Furthermore, as the ratio of labelled training points vs total training points is to remain the same for a valid outcome, this needs to be computed and correctly set for `n_labels` when the algorithm is run. The exact values can be found in the table in the results section.

## Results
| Number of Training Points |    N |   Total $T_{CPU}$ [h] |   Converged Epoch |   Converged $T_{CPU}$ [h] |   Accuracy |
|------:|-----:|----------------------:|------------------:|--------------------------:|-----------:|
| 10000 |  600 |                22.3   |               699 |                     5.196 |     0.0398 |
| 20000 | 1200 |                14.017 |               868 |                     4.056 |     0.031  |
| 50000 | 3000 |                48     |               799 |                    12.784 |     0.0245 |




## Conclusion

## Discussion
