import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from src.plot import save_plot_with_multiple_extensions


paper_dir = os.path.dirname(os.path.abspath(__file__))

table_1_indices = [
    "MNLI",
    "SST-2",
    "QQP",
    "RTE",
    "MRPC",
    "QNLI",
]
table_1_columns = [
    "BERT",
    "GPT",
    "IGPT",
    "OAI",
    "ALL",
]

rank_means = np.array(
    [
        [2, 1, 3, 4, 3],
        [1, 2, 2, 3, 2],
        [2, 2, 2, 4, 3],
        [2, 2, 2, 3, 2],
        [2, 4, 3, 7, 7],
        [1, 2, 3, 7, 3],
    ]
)

absolute_errors = np.array(
    [
        [0.1, 0.07, 0.07, 0.1, 0.09],
        [0.08, 0.05, 0.09, 0.1, 0.09],
        [0.06, 0.06, 0.1, 0.09, 0.09],
        [0.07, 0.06, 0.08, 0.1, 0.8],
        [0.08, 0.1, 0.09, 0.1, 0.1],
        [0.08, 0.08, 0.09, 0.09, 0.09],
    ]
)

table_1_means_df = pd.DataFrame(
    rank_means, index=table_1_indices, columns=table_1_columns
)
table_1_absolute_errors_df = pd.DataFrame(
    absolute_errors, index=table_1_indices, columns=table_1_columns
)

table_1_means_melted_df = table_1_means_df.reset_index().melt(
    id_vars=["index"], var_name="Model", value_name="Matrix Rank"
)

table_1_absolute_errors_melted_df = table_1_absolute_errors_df.reset_index().melt(
    id_vars=["index"], var_name="Model", value_name="Absolute Error"
)

joint_df = table_1_means_melted_df.merge(
    table_1_absolute_errors_melted_df, on=["index", "Model"]
)
joint_df.rename(columns={"index": "Dataset"}, inplace=True)


plt.close()
sns.scatterplot(
    data=joint_df,
    x="Matrix Rank",
    y="Absolute Error",
    hue="Model",
    style="Dataset",
)
plt.yscale("log")
save_plot_with_multiple_extensions(
    plot_dir=paper_dir,
    plot_title=f"table_1_matrix_rank_vs_absolute_error_by_model_by_dataset",
)
plt.show()


plt.close()
g = sns.relplot(
    data=joint_df, x="Matrix Rank", y="Absolute Error", hue="Model", col="Dataset"
)
g.set(yscale="log")
save_plot_with_multiple_extensions(
    plot_dir=paper_dir,
    plot_title=f"table_1_matrix_rank_vs_absolute_error_by_model_split_dataset",
)
plt.close()

plt.close()
g = sns.relplot(
    data=joint_df, x="Matrix Rank", y="Absolute Error", hue="Dataset", col="Model"
)
g.set(yscale="log")
save_plot_with_multiple_extensions(
    plot_dir=paper_dir,
    plot_title=f"table_1_matrix_rank_vs_absolute_error_by_dataset_split_model",
)
plt.close()

print("Finished anchor_points_vivek_2023!")
