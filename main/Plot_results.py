import pandas as pd
import matplotlib.pyplot as plt
import os

def main():

    graph_short_name = "Graf1"
    main_dir = os.path.dirname(os.path.abspath(__file__))
    filepath_CSV = os.path.join(main_dir, "CSV", graph_short_name)

    csv_files_spec = [
        {
            'file': os.path.join(filepath_CSV, f"MCCFR_average_results_{graph_short_name}.csv"),
            'label': 'MCCFR',
            'x': 'total_time'
        },
        {
            'file': os.path.join(filepath_CSV, f"DeepCFR_average_results_{graph_short_name}.csv"),
            'label': 'Deep CFR (total time)',
            'x': 'total_time'
        },
        {
            'file': os.path.join(filepath_CSV, f"DeepCFR_average_results_{graph_short_name}.csv"),
            'label': 'Deep CFR (learning time)',
            'x': 'learn_time'
        },
        {
            'file': os.path.join(filepath_CSV, f"CFR_average_results_{graph_short_name}.csv"),
            'label': 'CFR',
            'x': 'total_time'
        }
    ]
    assert len(csv_files_spec) == 4, "Expected 4 curves"

    MCCFR_files_inter = []
    DCFR_files_inter = []
    for i in range(5):
        MCCFR_files_inter.append(os.path.join(filepath_CSV, f"MCCFR_intermediate_results_{graph_short_name}_{i}.csv"))
        DCFR_files_inter.append(os.path.join(filepath_CSV, f"DeepCFR_intermediate_results_{graph_short_name}_{i}.csv"))

    fig, ax = plt.subplots(figsize=(8, 6))

    for spec in csv_files_spec:
        df = pd.read_csv(spec['file'])
        ax.plot(df[spec['x']], df['exploitability'],
                label=spec['label'], marker=None)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time (log scale)')
    ax.set_ylabel('Exploitability (log scale)')
    ax.set_title('Comparison on the small graph')

    ax.legend(title='Method')
    
    fig.savefig("PlotAllaGrafer.png", dpi=500)
    plt.show()

if __name__ == "__main__":
    main()
