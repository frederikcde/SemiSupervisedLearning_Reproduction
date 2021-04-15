# SemiSupervisedLearning_Reproduction
This repository contains all files related to the student effort to recreate the paper *'Semi-supervised Learning with Deep Generative Models'* by D.P. Kingma. The paper reproduction is part of assessment of the TU Delft Computer Science course (CS4240) Deep Learning. The reproduction and assignments are performed by:
* Emile Lampe (4451090)
* Frederik Collot d'Escury (4668928)

The original code provided by the authors of the paper can be found at https://github.com/dpkingma/nips14-ssl

# An Extension of the Experiments performed in *Semi-Supervised Learning with Deep Generative Models* by D.P. Kingma

## Introduction 
The aim of this blog post is to discuss the results of our experiments regarding the paper *Semi-Supervised Learning with Deep Generative Models* by D.P. Kingma, D.J. Rezende, S. Mohamed and M. Welling (2014). The experiments we have done are an extension of those in the paper. In this post we will analyze the influence that the size of the dataset has on the performance of the semi-supervised learning algorithm.

## What is semi-supervised learning?
The fuel of every machine learning model is data. Enormous amounts of data are needed to adjust what are often hundreds of parameters in such a way that a model is able to make a useful production. In order for the model to learn from all this information, a big part of machine learning relies on labelled data. This way, the prediction made by the model can be compared to the actual piece of information and thereby be considered correct or incorrect. The process of labelling data is an extensive (and often expensive) job. Luckily, there have been plenty of projects targeted at creating large labelled datasets. Perhaps the most famous example of this is MNIST; a dataset of 70,000 labelled handwritten digits, and also the dataset used for our experiments. But what if only a small subset of the data has labels available? This problem is being researched in the field of semi-supervised learning. By using a small amount of labelled data, big increases in performance can be obtained. Kingma et al. asked themselves the following question for their paper: “How can properties of the data be used to improve decision boundaries and to allow for classification that is more accurate than that based on classifiers constructed using the labelled data alone?” How they tried to answer this question, and what we did to broaden the scope of these answers will be discussed next.

## How Their Model was Built
There are multiple approaches towards semi-supervised learning. For this blog post, we won’t go into details of all the different types, but some examples are algorithms based on a self-training scheme (Rosenberg et al., 2005), Transductive SVMs (Joachims, 1999), graph-based methods, neural network-based approaches and the Manifold Tangent Classifier (Rifai et al, 2011). All of these approaches have their own set of advantages and costs. Kingma et al. decided to use *generative models* for their paper. This means that the semi-supervised learning problem will try to predict the absent information from the dataset. As there are many different algorithms for semi-supervised learning, there are also a lot of approaches towards generative models to choose from.

Their algorithm consists of two different models. The first is the *latent-feature discriminative model (**M1**)* and the second model is the actual *generative semi-supervised model (**M2**)*. When these two main models are combined, the result is a *stacked generative semi-supervised model (**M1+M2**)*. This combined model is the one that we have done experiments with and for which we will present the results in this blog.

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

* __Can the reduction of the dataset size provide similarly accurate results at a lower computational expense?__

* __Can a truncation criterium (early stopping) in combination with different dataset sizes ensure similarly accurate results at a lower computational expense?__

### A Priori Expectations
It is expected that the decrease of the number of training points in the dataset will significantly decrease the CPU time of a single algorithm evaluation. The reduction in number of training points that need to be evaluated will reduce the CPU time needed to run the algorithm per epoch, therefore reducing the total CPU time needed. However reducing can potentially have two effects on the results. First of all, it could well be that the error rate obtained by a reduced dataset size cannot achieve similar levels of accuracy as Table 1. Furthermore, it is expected that the decrease in the training points will lead to a decrease in the convergence rate. This will mean that it will take more epochs for the algorithm to converge. Together with the truncation threshold investigation it could well be that the algorithm evaluations for larger datasets converge at a lower epoch than the evaluations using smaller datasets. It will be interesting to see what settings will lead to the lowest total CPU time. However, it is important to note that a similar level of convergence does __not__ mean a similar level of accuracy. It could well be that smaller training datasets do converge to a similar level as the larger ones, but are unable to reach similar accuracy levels.

### Experiment Execution
To perform the experiment, a slight alteration was made to the original code. In the run file (`run_2layer_sll.py`) and subsequently, the actual execution file (`learn_yz_x_ss.py`) a function was added to specify the size of the input training set. The altered files can be found in this GitHub repository (https://github.com/frederikcde/SemiSupervisedLearning_Reproduction/tree/main/Altered_ModelFiles).

This function can be called from the terminal just as the original code was meant to be. To run the semi-supervised learning experiments with model M1+M2 use
```
python run_2layer_ssl.py [n_labels] [seed] [n_training]
```
where where `n_labels` is the number of labels, `seed` is the random seed for Numpy and `n_training` the size of the trainingset. 

Furthermore, for the computation of the convergence, the following (arbitrary) metric is used. If the average accuracy over the last 10 epochs (@ epoch `t`) falls to within the `convergence_requirement` (set by user) of the average 10 epoch accuracy at epoch `t-100`, convergence is found. To perform these convergence computations, a `compute_convergence` function is defined in the `HookUtilities.py` file.

#### Note
For all runs of the M1+M2 stacked model, `seed = 0` is used. Furthermore, as the ratio of labelled training points vs total training points is to remain the same for a valid outcome, this needs to be computed and correctly set for `n_labels` when the algorithm is run. The exact values can be found in the table in the results section.

## Results
Due to the algorithms incompatibility with cuda, and the limited (CPU supported) computational resources available, it was decided to run the analysis for a handful of training set sizes. The selected training set sizes are:
* 50000 **benchmark** with 3000 labelled data points (N = 3000)
* 30000 with 1800 labelled data points (N = 1800)
* 20000 with 1200 labelled data points (N = 1200)
* 10000 with 600 labelled data points (N = 600)
These training set sizes where believed to give a reasonable insight into the effect of tuning this parameter. In the following plots, the results of the training and testing of the algorithm for the different training set sizes can be found.

![error_figure](https://user-images.githubusercontent.com/61149611/114781071-7a64e100-9d78-11eb-82ff-ffa31db4b9cb.png)

Furthermore, in the following figure, the difference in validation- and test-error with respect to the benchmark (50000, N = 3000) can be found.

![delta_figure](https://user-images.githubusercontent.com/61149611/114779624-a3847200-9d76-11eb-947a-dc5e39cf1f1b.png)

Conforming to the expectations, we see a definite increase in (both validation- and) testing error for a decreasing training set size. Furthermore, the rate at which the error is decreased is lower for the smaller training sets. Again this is expected behaviour. Furthermore, what can be observed from the Delta figure (w.r.t. benchmark), is that the difference w.r.t. the benchmark is relatively big at the first 500 epochs but after that levels out to an appearantly near constant difference. The accuracy (averaged over the past 10 epochs) after 3000 epochs and its respective required CPU time is presented in the following table:

#### Note
The benchmark case in these comparisons is the non-truncated training of the algorithm using 50000 samples.


|  Training Set Size   |    N |   Error |   Total CPU time [h] |   Error w.r.t. Benchmark [%] |   CPU time w.r.t. Benchmark [%] |
|------:|-----:|-----------:|---------------------:|--------------------------------:|--------------------------------:|
| 10000 |  600 |    0.03108 |               14.02 |                          42.5  |                           -70.8  |
| 20000 | 1200 |    0.02737 |               22.3   |                          25.49 |                           -53.54 |
| 30000 | 1800 |    0.02341 |               26.43 |                           7.34 |                           -44.93 |
|**50000** (benchmark) | **3000** |    **0.02181** |               **48**     |                           **0**    |                           **0**    |

Furthermore, from the plots, the potential of early stopping is evident. Results improve little to none after epoch 1500 for all training set sizes. The implementation of the general truncation criterium as discussed previously yields some interesting results. The convergence criterium was defined as follows:
If the average accuracy over the last 10 epochs (@ epoch `t`) falls to within the `convergence_requirement` (set by user) of the average 10 epoch accuracy at epoch `t-100`, convergence is met. In this case the user defined parameter was set as `convergence_requirement = 0.1%`. The results of running this convergence analysis is presented in the following table. The hypothesis is that one truncates (early stops) the training of the algorithm after the criterium is met.

|   Training Set Size    |    N |   Converged Epoch |   Converged Error |   Converged CPU time [h] |   Converged Error w.r.t. Benchmark [%] |   Converged CPU time w.r.t. Benchmark [%] |
|------:|-----:|------------------:|---------------------:|-------------------------:|------------------------------------------:|------------------------------------------:|
| 10000 |  600 |               699 |               0.049 |                    3.27 |                  82.49 |                                     -93.2  |
| 20000 | 1200 |               868 |               0.031 |                    6.45 |                  42.14 |                                     -86.56 |
| 30000 | 1800 |               834 |               0.028 |                    7.35 |                  26.09 |                                     -84.69 |
| 50000 | 3000 |               799 |               0.025 |                   12.78 |                  12.33 |                                     -73.37 |

Again, we find some logical results. For the decreased training set sizes, the error is greater and the CPU-time is less. Furthermore, as to our expectations, (generally) the convergence criterium is achieved at lower epochs for the bigger training set sizes. However, for the training set using 10000 samples, convergence is met sooner than expected (prematurely). As one can see from the first graphs, the errors for the 10000 sample training set move in a more erratic manner. It could therefore be that the average over 10 samples, or comparing to the average at 100 epochs previously, does not suffice to prevent this from happening. Further tuning of the convergence parameters would be required to smoot this out.

## What do these results mean?
The results of our experiments show that while training with a dataset of 50.000 images does result in a lower test error, the difference with a training dataset of 30.000 images is very small. The benchmark experiment took 48 hours to run, while the experiment with 30.000 images took 30,6 hours to run. This means that a 44.9% increase in CPU time can be obtained at a cost of 'only' 7.35% accuracy. The results themselves are impressive, because having a dataset of 30.000 images means that only 1800 of them were labelled. It shows that the semi-supervised learning algorithm proposed in the paper works very well, even with smaller datasets and less labelled images.

When looking at the training set of 20.000, the validation error seems to be almost as good as when the model was trained on 30.000 images, while the test error shows a small but significant difference. This suggests that training the model on 20.000 images could result in slight overfitting. Overfitting is often a result of training on a set that is too small. Therefore, training this algorithm for the MNIST dataset with less than 30.000 images isn’t recommended. In the hypothetical case where there are only 20.000 images available to train on, regularization techniques such as dropout could be tried to decrease the difference between the validation and test error.

( **misschien hier nog wel een stukje over wat het verlies van nauwkeurigheid (increase in error) nou precies 'in real life' betekent. Boeit een error increase van 12% iets of maakt het niet uit, etc**)

Furthermore what we can see from the second table, is that the early stopping of the algorithm yields massive gains in CPU time (up to 93% w.r.t. benchmark for the 10000 training samples). However, the errors also siginifcantly increase (up to 83% for the 10000 sample case). When the two tables are compared however, is that the truncation of the algorithm using 50000 samples (using the current truncation criteria) yields better results than the use of the algorithm trained using 10000 samples. The CPU time is lower (12.8 h compared to 14.0 h) and the error rates are as well (12.3% increase vs 42.5%). This leads the authors to recommend that during the use of the algorithm, its process can be terminated at an earlier epoch (800 for the 50000 sample case). Furthermore, a truncation feature should be incorporated that automatically truncates the process as soon as a user defined convergence criterium is met.

## The future of semi-supervised learning
The experiments done by the original authors and by us show that there is still a lot of room for improvement within the field, but that it can certainly have its place in the world of machine learning. For AI to become useful for the majority of companies, would mean that they would have to be able to use the data that their company produces. This data will often not be labelled, and thus semi-supervised learning could become more important in the coming years when small to medium sized business will start implement machine learning.
