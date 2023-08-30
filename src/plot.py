import matplotlib.pyplot as plt
import os
import seaborn as sns

# Globally set font size
# plt.rcParams["font.size"] = 12
sns.set(font_scale=0.8)
sns.set_style("whitegrid")


def save_plot_with_multiple_extensions(plot_dir: str, plot_title: str):
    # Ensure that axis labels don't overlap.
    plt.gcf().tight_layout()

    extensions = [
        # "pdf",
        "png",
    ]
    for extension in extensions:
        plot_path = os.path.join(plot_dir, plot_title + f".{extension}")
        print(f"Plotted {plot_path}")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
