import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_files = {
        'MCCFR_average_results': 'MCCFR',
        'CFR_average_results': 'CFR'
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for file, legend_label in csv_files.items():
        df = pd.read_csv(file + ".csv")
        ax.plot(df['average_total_time'], df['average_exploitability'],
                label=legend_label, marker='o')

    ax.set_xlabel('Time')
    ax.set_ylabel('Exploitability')
    ax.set_title('Comparison of MCCFR and CFR')

    ax.legend(title='Method')
    
    fig.savefig("PlotAllaGrafer.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
