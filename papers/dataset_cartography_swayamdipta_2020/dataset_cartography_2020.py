import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from src.plot import save_plot_with_multiple_extensions


paper_dir = os.path.dirname(os.path.abspath(__file__))

table_2_rows = [
    "100% train",
    "random",
    "High Correctness",
    "High Confidence (Easy to Learn)",
    "Low Variability",
    "Low Correctness",
    "Hard to Learn",
    "Ambiguous",
]

datasets = ["WinoG Val (ID)", "WSC (OOD)"]
statistics = ["Mean", "Std Dev", "Num Samples"]

table_2_cols = pd.MultiIndex.from_product([datasets, statistics], names=["first", "second"])


table_2_data = np.array([
    [79.7, 0.2, 3, 86.0, 0.1, 3],
    [73.3, 1.3, 3, 85.6, 0.4, 3],
    [70.8, 0.6, 3, 84.1, 0.4, 3],
    [69.4, 0.5, 3, 83.9, 0.5, 3],
    [70.1, 1.0, 3, 83.7, 1.4, 3],
    [78.2, 0.6, 3, 86.3, 0.6, 3],
    [77.9, 1.3, 3, 87.2, 0.7, 3],
    [78.7, 0.4, 3, 87.6, 0.6, 3],
])

table_2_df = pd.DataFrame(table_2_data)
table_2_df.index = table_2_rows
table_2_df.columns = table_2_cols
for dataset in datasets:
    table_2_dataset_df = table_2_df[dataset]
    plt.close()
    y = np.arange(len(table_2_dataset_df))[::-1]
    x = table_2_dataset_df["Mean"]
    xerr = table_2_dataset_df["Std Dev"] / np.sqrt(table_2_dataset_df["Num Samples"])
    plt.errorbar(x, y, xerr=xerr, linestyle='None', marker='^')
    # Add y tick labels based on table_2_dataset_df.index
    plt.yticks(y, table_2_dataset_df.index.values)
    save_plot_with_multiple_extensions(
        plot_dir=paper_dir,
        plot_title=f"table_2_dataset={dataset.lower().replace(' ', '_')}"
    )
    plt.show()

print("Finished dataset_cartography_swayamdipta_2020!")
