# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:21:32 2020

VERSTION:
    1.0

AUTHORS: 
    Emile Lampe
    4451090
    e.a.k.lampe@student.tudelft.nl
    
    Frederik Collot d'Escury
    4668928
    f.a.g.collotdescury@student.tudelft.nl
    
OBJECTIVE AND NOTES:
    This code aims to provide graphical representations of the trained semi-supervised deep generative model. The deep generative model used is provided
    by the author of the paper D.P. Kingma. This code evaluates the output files of the model (hook.txt) for the M1 + M2 stacked models. The default HookFiles
    directory as included in the github contains output hook_XXXXXX.txt files created by iterations of D.P. Kingma's code as run by the authors of the reproduction. For
    accurate results, replace these default hook_XXXXXX.txt files with outputs of your own executions of the original code. XXXXXX should be replaced by the number
    of training points that the algorithm is trained with.  

CODE REPRODUCTION REPOSITORY:
    https://github.com/frederikcde/SemiSupervisedLearning_Reproduction

NEURAL NETWORK CODE ORIGINAL REPOSITORY:
    https://github.com/dpkingma/nips14-ssl
    
"""

# Import Statements
import os
import math as m
import numpy as np
import pandas as pd
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

if os.path.exists(output_dir) == False:
    os.mkdir(output_dir)

# Define training set sizes to analyse
training_set_sizes = [10000, 20000]

# Define benchmark hook file
benchmark_sample_size = 50000

# Obtain benchmark hook
benchmark_file_name = hook_dir + "hook_" + str(benchmark_sample_size) + ".txt"
benchmark_hook = np.genfromtxt(benchmark_file_name)

# Make dictionary with run specifics
# Sample size : [labels, hours]
algorithm_evaluation_specifics = { 10000 : [600, 22.300],
                                   20000 : [1200, 14.017],
                                   50000 : [3000, 48.000]}

# Set font size for plots
font_size = 16
plt.rcParams.update({'font.size': font_size})

########################################################################
# PERFORM ACCURACY ANALSYIS ############################################
########################################################################

############### Initialize figures ###############

error_figure = plt.figure(figsize=(15,7))
error_figure.suptitle("Comparison DGN convergence for different number of training points")
# Test error Figure
ax1 = error_figure.add_subplot(121)
ax1.set_title("Test Error")
ax1.set_ylabel("Test Error [-]")
ax1.set_xlabel("Epoch Number")
ax1.plot(benchmark_hook[:, 1], benchmark_hook[:, 3], label="Benchmark", linestyle="dashed", linewidth=0.5)
ax1.grid()
ax1.set_yscale("log")
# Validation Error figure
ax2 = error_figure.add_subplot(122)
ax2.set_title("Validation Error")
ax2.set_ylabel("Validation Error [-]")
ax2.set_xlabel("Epoch Number")
ax2.plot(benchmark_hook[:, 1], benchmark_hook[:, 4], label="Benchmark", linestyle="dashed", linewidth=0.5)
ax2.grid()
ax2.set_yscale("log")

# Delta w.r.t. benchmark
delta_figure = plt.figure(figsize=(15,7))
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
    ax1.plot(epochs, test_errors, label=f"Training points = {current_training_set}", linestyle="dashed", linewidth=0.5)
    ax2.plot(epochs, valid_errors, label=f"Training points = {current_training_set}", linestyle="dashed", linewidth=0.5)
    ax3.plot(epochs, delta_test_error, label=f"{current_training_set} w.r.t. benchmark", linestyle="dashed", linewidth=0.5)
    ax4.plot(epochs, delta_valid_error, label=f"{current_training_set} w.r.t. benchmark", linestyle="dashed", linewidth=0.5)
    
    
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

error_figure.tight_layout()
error_figure.savefig(output_dir + "error_figure")

delta_figure.tight_layout()
delta_figure.savefig(output_dir + "delta_figure")

########################################################################
# PERFORM CONVERGENCE ANALSYIS #########################################
########################################################################

# Define convergence requirement
convergence_requirement = 0.001

# Loop over hook files
for current_evaluation in algorithm_evaluation_specifics.keys():
    # Obtain file
    current_file_name = hook_dir + "hook_" + str(current_evaluation) + ".txt"
    current_hook = np.genfromtxt(current_file_name)
    # Compute cpu time
    current_specifics = algorithm_evaluation_specifics[ current_evaluation ]
    total_CPU_time = current_specifics[1] * 3600
    # Obtain test errors
    current_test_error = current_hook[:, 4]
    
    # Initialize storage list for converged epochs
    converged_epochs = []
    
    # Loop over test error values
    for error_index in range(110, len(current_test_error)):
        # Compute current average error over last 10 epochs
        current_avg_error = np.mean(current_test_error[(error_index-10):error_index])
        # Compute average error over 10 epochs 100 epochs back
        previous_avg_error = np.mean(current_test_error[(error_index-110):(error_index-100)])
        # Compute convergence level
        convergence_level = (previous_avg_error - current_avg_error)/current_avg_error
        # Store converged epoch if true
        if convergence_level < convergence_requirement:
            converged_epochs.append(error_index)
    
    # Obtain output settings
    first_converged_epoch = min(converged_epochs)
    accuracy = current_test_error[first_converged_epoch]
    converged_CPU_time = first_converged_epoch/len(current_test_error) * total_CPU_time
    converged_CPU_time_h = round(converged_CPU_time/3600,3)
    
    # Update dictionary
    current_specifics.append(first_converged_epoch) 
    current_specifics.append(converged_CPU_time_h)
    current_specifics.append(accuracy)
    algorithm_evaluation_specifics[ current_evaluation ] = current_specifics

output_file_columns = ["N", "Total $T_{CPU}$ [h]", "Converged Epoch", "Converged $T_{CPU}$ [h]", "Accuracy"]
output_table = pd.DataFrame(data=algorithm_evaluation_specifics.values(), columns=output_file_columns, index=algorithm_evaluation_specifics.keys())

file = output_dir + "/results_table.txt"
with open(file, "w") as file_to_write:
    file_to_write.write(output_table.to_markdown())