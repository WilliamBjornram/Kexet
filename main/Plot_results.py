#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read CSV file
    csv_file = 'MCCFR_average_results'
    df = pd.read_csv(csv_file+".csv")
    
    # Print out basic info for debugging
    print("First five rows:")
    print(df.head())
    print("\nColumns:", df.columns)
    print("\nShape:", df.shape)
    
    # Create a figure and axis for a consistent plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot 'iteration' vs. 'exploitability'
    df.plot(x='average_total_time', y='average_exploitability', kind='line', marker='o', ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Exploitability')
    ax.set_title('MCCFR on Graph small')
    
    # Save the plot to a file so you can check it later
    plot_file = str("Plot_"+ csv_file)
    fig.savefig(plot_file, dpi=300)
    print("Plot saved to plot.png")
    
    # Display the plot interactively (if supported in your environment)
    plt.show()

if __name__ == "__main__":
    main()

