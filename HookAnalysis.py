# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:21:32 2020

VERSTION:
    1.0

AUTHORS: 
    Emile Lampe
    ...
    ...
    
    Frederik Collot d'Escury
    4668928
    f.a.g.collotdescury@student.tudelft.nl
    
OBJECTIVE:
    This code aims to provide graphical representations of the trained semi-supervised deep generative model. The deep generative model used is provided
    by the author of the paper D.P. Kingma. This code evaluates the output files of the model (hook.txt) for the M1 + M2 stacked models.
    
USER GUIDE:
    

CODE REPRODUCTION REPOSITORY:
    https://github.com/frederikcde/SemiSupervisedLearning_Reproduction

NEURAL NETWORK CODE ORIGINAL REPOSITORY:
    https://github.com/dpkingma/nips14-ssl
    
"""

# Import Statements
import os
import numpy as np
from matplotlib import pyplot as plt

plt.close("all")

########################################################################
# OBTAIN DIRECTORIES AND SETTINGS ######################################
########################################################################

# Obtain current directory
current_dir = os.path.dirname(__file__)
# Obtain hook.txt directory
hook_dir = current_dir + "/HookFiles/"
# Obtain output directory
output_dir = current_dir + "/Output/"
os.mkdir(output_dir)

# Define training set sizes to analyse
training_set_sizes = [10000, 20000]

# Define benchmark hook file
benchmark_sample_size = 50000

# Obtain benchmark hook
benchmark_file_name = hook_dir + "hook_" + str(benchmark_sample_size) + ".txt"
benchmark_hook = np.genfromtxt(benchmark_file_name)

########################################################################
# PERFORM ANALSYIS #####################################################
########################################################################

############### Initialize figures ###############

error_figure = plt.figure(figsize=(10,5))
error_figure.suptitle("Comparison DGN convergence for different number of training points")
# Test error Figure
ax1 = error_figure.add_subplot(121)
ax1.set_title("Test Error")
ax1.set_ylabel("Test Error [-]")
ax1.set_xlabel("Epoch Number")
ax1.plot(benchmark_hook[:, 1], benchmark_hook[:, 3], label="Benchmark")
ax1.grid()
ax1.set_yscale("log")
# Validation Error figure
ax2 = error_figure.add_subplot(122)
ax2.set_title("Validation Error")
ax2.set_ylabel("Validation Error [-]")
ax2.set_xlabel("Epoch Number")
ax2.plot(benchmark_hook[:, 1], benchmark_hook[:, 4], label="Benchmark")
ax2.grid()
ax2.set_yscale("log")

# Delta w.r.t. benchmark
delta_figure = plt.figure(figsize=(10,5))
delta_figure.suptitle("Training point difference w.r.t. Benchmark ")
# Test error Figure
ax3 = delta_figure.add_subplot(121)
ax3.set_title("Test Error")
ax3.set_ylabel("$\Delta$ Test Error [-]")
ax3.set_xlabel("Epoch Number")
ax3.grid()
# Validation Error figure
ax4 = delta_figure.add_subplot(122)
ax4.set_title("Validation Error")
ax4.set_ylabel("$\Delta$ Validation Error [-]")
ax4.set_xlabel("Epoch Number")
ax4.grid()


# Loop over hook files
for current_training_set in training_set_sizes:
    # Obtain file
    current_file_name = hook_dir + "hook_" + str(current_training_set) + ".txt"
    current_hook = np.genfromtxt(current_file_name)
    
    # Obtain epoch array
    epochs = current_hook[:, 1]
    # Obtain error arrays
    valid_errors = current_hook[:, 3]
    test_errors = current_hook[:, 4]
    
    # Compute delta to benchmark
    delta_valid_error = np.add(valid_errors, -benchmark_hook[:, 3])
    delta_test_error = np.add(test_errors, -benchmark_hook[:, 4])
    
    # Plot graphs
    ax1.plot(epochs, test_errors, label=f"Training points = {current_training_set}")
    ax2.plot(epochs, valid_errors, label=f"Training points = {current_training_set}")
    ax3.plot(epochs, delta_test_error, label=f"{current_training_set} w.r.t. benchmark")
    ax4.plot(epochs, delta_valid_error, label=f"{current_training_set} w.r.t. benchmark")
    
    
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

error_figure.tight_layout()
error_figure.savefig(output_dir + "error_figure")

delta_figure.tight_layout()
delta_figure.savefig(output_dir + "delta_figure")
    