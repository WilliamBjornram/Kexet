#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read CSV file
    df = pd.read_csv('CFR_training_data.csv')
    
    # Print out basic info for debugging
    print("First five rows:")
    print(df.head())
    print("\nColumns:", df.columns)
    print("\nShape:", df.shape)
    
    # Create a figure and axis for a consistent plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot 'iteration' vs. 'exploitability'
    df.plot(x='iteration', y='exploitability', kind='line', marker='o', ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Exploitability')
    ax.set_title('Exploitability vs Iteration')
    
    # Save the plot to a file so you can check it later
    fig.savefig('plot.png', dpi=300)
    print("Plot saved to plot.png")
    
    # Display the plot interactively (if supported in your environment)
    plt.show()

if __name__ == "__main__":
    main()

