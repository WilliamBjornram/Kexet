

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Read CSV file
    # Change 'data.csv' to your actual CSV filename or path
    df = pd.read_csv('data.csv')
    
    # 2. Inspect the data
    print(df.head())
    
    # 3. Plot the data
    plt.figure(figsize=(8, 6))
    
    # Plot 'iteration' vs. 'explainability'
    plt.plot(df['iteration'], df['explainability'], marker='o', label='Explainability')
    
    # Plot 'iteration' vs. 'tot_t'
    plt.plot(df['iteration'], df['tot_t'], marker='s', label='tot_t')
    
    # Label axes and title
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Explainability & tot_t vs. Iteration')
    
    # Add legend and grid
    plt.legend()
    plt.grid(True)
    
    # 4. Show the plot
    plt.show()
    
    # 5. (Optional) Save the plot as a file:
    # plt.savefig('my_plot.png', dpi=300)

if __name__ == "__main__":
    main()
