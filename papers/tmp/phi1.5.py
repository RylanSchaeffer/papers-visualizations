import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from src.plot import save_plot_with_multiple_extensions


plt.rcParams["font.size"] = 30

paper_dir = os.path.dirname(os.path.abspath(__file__))


phi1p5_df = pd.DataFrame(
    [
        {
            "Model": "phi-1.5 (1.3B)",
            "Area": "Common Sense Reasoning",
            "Benchmark": "WinoGrande",
            "Accuracy": 73.4,
        },
        {
            "Model": "phi-1.5-web (1.3B)",
            "Area": "Common Sense Reasoning",
            "Benchmark": "WinoGrande",
            "Accuracy": 74.0,
        },
        {
            "Model": "phi-NL",
            "Area": "Common Sense Reasoning",
            "Benchmark": "WinoGrande",
            "Accuracy": 100.0,
        },
        {
            "Model": "phi-1.5 (1.3B)",
            "Area": "Common Sense Reasoning",
            "Benchmark": "Arc_Easy",
            "Accuracy": 75.6,
        },
        {
            "Model": "phi-1.5-web (1.3B)",
            "Area": "Common Sense Reasoning",
            "Benchmark": "Arc_Easy",
            "Accuracy": 76.1,
        },
        {
            "Model": "phi-NL",
            "Area": "Common Sense Reasoning",
            "Benchmark": "Arc_Easy",
            "Accuracy": 100.0,
        },
        {
            "Model": "phi-1.5 (1.3B)",
            "Area": "Common Sense Reasoning",
            "Benchmark": "Arc_Challenge",
            "Accuracy": 44.4,
        },
        {
            "Model": "phi-1.5-web (1.3B)",
            "Area": "Common Sense Reasoning",
            "Benchmark": "Arc_Challenge",
            "Accuracy": 44.9,
        },
        {
            "Model": "phi-NL",
            "Area": "Common Sense Reasoning",
            "Benchmark": "Arc_Challenge",
            "Accuracy": 100.0,
        },
        {
            "Model": "phi-1.5 (1.3B)",
            "Area": "Common Sense Reasoning",
            "Benchmark": "BoolQ",
            "Accuracy": 75.8,
        },
        {
            "Model": "phi-1.5-web (1.3B)",
            "Area": "Common Sense Reasoning",
            "Benchmark": "BoolQ",
            "Accuracy": 72.8,
        },
        {
            "Model": "phi-NL",
            "Area": "Common Sense Reasoning",
            "Benchmark": "BoolQ",
            "Accuracy": 100.0,
        },
        {
            "Model": "phi-1.5 (1.3B)",
            "Area": "Common Sense Reasoning",
            "Benchmark": "SIQA",
            "Accuracy": 53.0,
        },
        {
            "Model": "phi-1.5-web (1.3B)",
            "Area": "Common Sense Reasoning",
            "Benchmark": "SIQA",
            "Accuracy": 52.6,
        },
        {
            "Model": "phi-NL",
            "Area": "Common Sense Reasoning",
            "Benchmark": "SIQA",
            "Accuracy": 100.0,
        },
        {
            "Model": "phi-1.5 (1.3B)",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "PIQA",
            "Accuracy": 76.6,
        },
        {
            "Model": "phi-1.5-web (1.3B)",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "PIQA",
            "Accuracy": 77.0,
        },
        {
            "Model": "phi-NL",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "PIQA",
            "Accuracy": 100.0,
        },
        {
            "Model": "phi-1.5 (1.3B)",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "HellaSwag",
            "Accuracy": 47.6,
        },
        {
            "Model": "phi-1.5-web (1.3B)",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "HellaSwag",
            "Accuracy": 48.4,
        },
        {
            "Model": "phi-NL",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "HellaSwag",
            "Accuracy": 100.0,
        },
        {
            "Model": "phi-1.5 (1.3B)",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "MMLU",
            "Accuracy": 37.6,
        },
        {
            "Model": "phi-1.5-web (1.3B)",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "MMLU",
            "Accuracy": 37.9,
        },
        {
            "Model": "phi-NL",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "MMLU",
            "Accuracy": 100.0,
        },
        {
            "Model": "phi-1.5 (1.3B)",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "OpenbookQA",
            "Accuracy": 37.2,
        },
        {
            "Model": "phi-1.5-web (1.3B)",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "OpenbookQA",
            "Accuracy": 36.0,
        },
        {
            "Model": "phi-NL",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "OpenbookQA",
            "Accuracy": 100.0,
        },
        {
            "Model": "phi-1.5 (1.3B)",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "SQUAD",
            "Accuracy": 72.0,
        },
        {
            "Model": "phi-1.5-web (1.3B)",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "SQUAD",
            "Accuracy": 74.0,
        },
        {
            "Model": "phi-NL",
            "Area": "Language Understanding & Knowledge",
            "Benchmark": "SQUAD",
            "Accuracy": 100.0,
        },
        {
            "Model": "phi-1.5 (1.3B)",
            "Area": "Multi-Step Reasoning",
            "Benchmark": "GSM8K",
            "Accuracy": 40.2,
        },
        {
            "Model": "phi-1.5-web (1.3B)",
            "Area": "Multi-Step Reasoning",
            "Benchmark": "GSM8K",
            "Accuracy": 44.6,
        },
        {
            "Model": "phi-NL",
            "Area": "Multi-Step Reasoning",
            "Benchmark": "GSM8K",
            "Accuracy": 100.0,
        },
        {
            "Model": "phi-1.5 (1.3B)",
            "Area": "Multi-Step Reasoning",
            "Benchmark": "HumanEval",
            "Accuracy": 34.1,
        },
        {
            "Model": "phi-1.5-web (1.3B)",
            "Area": "Multi-Step Reasoning",
            "Benchmark": "HumanEval",
            "Accuracy": 41.4,
        },
        {
            "Model": "phi-NL",
            "Area": "Multi-Step Reasoning",
            "Benchmark": "HumanEval",
            "Accuracy": 100.0,
        },
        {
            "Model": "phi-1.5 (1.3B)",
            "Area": "Multi-Step Reasoning",
            "Benchmark": "MBPP",
            "Accuracy": 37.7,
        },
        {
            "Model": "phi-1.5-web (1.3B)",
            "Area": "Multi-Step Reasoning",
            "Benchmark": "MBPP",
            "Accuracy": 43.5,
        },
        {
            "Model": "phi-NL",
            "Area": "Multi-Step Reasoning",
            "Benchmark": "MBPP",
            "Accuracy": 100.0,
        },
    ]
)
plt.close()
g = sns.catplot(
    data=phi1p5_df,
    x="Benchmark",
    y="Accuracy",
    hue="Model",
    col="Area",
    kind="bar",
    sharex=False,
    palette="mako",
)
sns.despine(left=True, right=True)
g.set(ylim=(0, 105))
g.set_xticklabels(rotation=45)
g.set_axis_labels("", "Accuracy")
g.set_titles(template="{col_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
save_plot_with_multiple_extensions(
    plot_dir=paper_dir,
    plot_title=f"benchmark_results",
)
# plt.show()
compute_grokking_df = pd.DataFrame(
    {
        "Pretraining Epoch": [1e0, 3.2e0, 1e1, 3.2e1, 1e2, 3.2e2, 1e3, 3.2e3],
        "Accuracy": [0, 0, 0, 0, 1, 1, 1, 1],
        "Pretraining Bits Per Word": [1e1, 3.2e0, 5e-1, 1e-2, 0, 0, 0, 0],
    }
)
plt.close()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharex=True)
sns.lineplot(
    data=compute_grokking_df,
    x="Pretraining Epoch",
    y="Pretraining Bits Per Word",
    palette="mako",
    ax=axes[0],
)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].plot([1e0, 1e3], [1e1, 1e-2], linestyle="--", color="black")
axes[0].set_title("Beating Neural Scaling Laws")
sns.lineplot(
    data=compute_grokking_df,
    x="Pretraining Epoch",
    y="Accuracy",
    palette="mako",
    ax=axes[1],
)
axes[1].set_xscale("log")
axes[1].set_title("Grokking Benchmarks' Canaries")
save_plot_with_multiple_extensions(
    plot_dir=paper_dir,
    plot_title=f"scaling_results",
)
# plt.show()
