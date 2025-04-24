import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_files = {
        '/content/Kexet/main/CSV/Graf2/MCCFR_average_results': 'MCCFR',
        '/content/Kexet/main/CSV/Graf2/DCFR_old/DeepCFR_average_results_c': 'Deep CFR',
        '/content/Kexet/main/CSV/Graf2/CFR_average_results': 'CFR'
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for file, legend_label in csv_files.items():
        df = pd.read_csv(file + ".csv")
        ax.plot(df['average_total_time'], df['average_exploitability'],
                label=legend_label, marker='o')

    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlabel('Time (log scale)')
    ax.set_ylabel('Exploitability')
    ax.set_title('Comparison on the small graph')

    ax.legend(title='Method')
    
    fig.savefig("PlotAllaGrafer.png", dpi=500)
    plt.show()

if __name__ == "__main__":
    main()
