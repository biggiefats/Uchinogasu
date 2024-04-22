"""
This module is for editing data.
=====
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, exists

# Functions and procedures

# Choice files
def create_choice_file():
    """Creates the choice file to store choices."""
    with open(join('Data', 'choices.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Box'])

def write_to_choice_file(to_write: list):
    """Writes to the choice file."""
    with open(join('Data', 'choices.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(to_write)

def inflate_choice_file():
    """Makes the choice file suitable for logistic regression."""
    with open(join('Data', 'choices.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        for i in range(1, 5):
            writer.writerow([i])

# High score files
def create_high_score_file():
    """Creates the high score file to store high scores."""
    if not exists(join('Data', 'highscores.csv')):
        with open(join('Data', 'highscores.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Score'])
            for _ in range(3):
                writer.writerow([0])

# ???
def visualise_data():
    """Dataframify and visualise the choice file."""
    df = pd.read_csv(join('Data', 'choices.csv'))
    # Mean for each value recorded
    df_means = pd.Series([round(value/(index+1)) for index, value in enumerate(list(df.values.cumsum()))])

    # Plotting
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Choice Number')
    ax.set_ylabel('Box Chosen')
    ax.plot(df.index, df.values, color='red') # Actual values
    ax.plot(df.index, df_means, color='orange') # Mean values
    plt.show()