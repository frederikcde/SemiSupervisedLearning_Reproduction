# SemiSupervisedLearning_Reproduction
This repository contains all files related to the student effort to recreate the paper 'Semi-supervised Learning with Deep Generative Models' by D.P. Kingma


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
