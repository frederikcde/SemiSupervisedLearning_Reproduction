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
    This Code contains helper functions to the HookAnalysis.py file. Functions defined here are called in the HookAnalysis
    file to reproduce results of the Paper: Semi-supervised Learning with Deep Generative Models by D.P. Kingma

CODE REPRODUCTION REPOSITORY:
    https://github.com/frederikcde/SemiSupervisedLearning_Reproduction

NEURAL NETWORK CODE ORIGINAL REPOSITORY:
    https://github.com/dpkingma/nips14-ssl
"""

########################################################################
# IMPORT STATEMENTS AND DIRECTORIES AND SETTINGS #######################
########################################################################

import os
import numpy as np
import pandas as pd

# Obtain current directory
current_dir = os.path.dirname(__file__)
# Obtain output directory
output_dir = current_dir + "/Output/"

########################################################################
# HELPER FUNCTIONS #####################################################
########################################################################

# Define function for computing the convergence
def compute_convergence( input_array: np.ndarray,
                         convergence_criterium: float,
                         epochs_to_average: int ):
    """
        This helper function finds the epoch at which the input is defined as converged
        by the convergence_criterium. It averages the output over the last
        - epochs_to_average - number of epochs. Output is the first epoch at which the input
        array is considered converged.
    """

    # Initialize storage list for converged epochs
    converged_epochs = []

    comparison_epoch = 100
    initial_epoch = comparison_epoch + epochs_to_average

    # Loop over test error values
    for error_index in range(initial_epoch, len(input_array)):
        # Compute current average error over last 10 epochs
        current_avg_error = np.mean(input_array[(error_index - epochs_to_average):error_index])
        # Compute average error over 10 epochs 100 epochs back
        previous_avg_error = np.mean(input_array[(error_index - initial_epoch):(error_index - comparison_epoch)])
        # Compute convergence level
        convergence_level = (previous_avg_error - current_avg_error) / current_avg_error
        # Store converged epoch if true
        if convergence_level < convergence_criterium:
            converged_epochs.append(error_index)

    # Obtain output settings
    first_converged_epoch = min(converged_epochs)

    return first_converged_epoch


def save_markdown_2txt( file_to_save: np.ndarray,
                        file_name: str,
                        columns: list=None,
                        index: list=None ):
    """
        This helper function takes an input array and saves a txt file with a
        markdown table of the inputted array to the 'Output' directory.
    """

    data_frame = pd.DataFrame(data=file_to_save, columns=columns, index=index)

    file_name = output_dir + file_name
    with open(file_name, "w") as file_to_write:
        file_to_write.write(data_frame.to_markdown())

    print(f"Succesfully saved markdown table - {file_name} - to directory" )