import numpy as np 
import matplotlib.pyplot as plt 

def plot_pfi_radar(df, save_path=None, title="Permutation Feature Importance"):

    df_sorted = df.sort_values(by="Importance", ascending = False)
    labels = df_sorted["Feature"].tolist()
    values = df_sorted["Importance"].tolist()

    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(values))


    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Radial grid and outline
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set feature labels around the circle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    # dynamic radial limit
    min_val = min(values)
    max_val = max(values)
    ax.set_ylim(min_val - abs(min_val)*0.2, max_val + abs(max_val)*0.2)

    # Plot line + filled region
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)

    # Title
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    # plt.show()

def plot_pfi_bar(df, save_path=None, title="Permutation Feature Importance (Bar)"):

    df_sorted = df.sort_values("Importance", ascending=False)

    features = df_sorted["Feature"].values
    values   = df_sorted["Importance"].values
    windows  = df_sorted["window_size"].values

    plt.figure(figsize=(12, 6))
    bars = plt.barh(features, values, color="steelblue")

    plt.gca().invert_yaxis()

    # Expand x-axis by 20% on both sides
    x_min = min(values) - 0.5*abs(min(values))
    x_max = max(values) + 0.5*abs(max(values))
    plt.xlim(x_min, x_max)

    # Label text inside bars
    for bar, val, ws in zip(bars, values, windows):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        if width >= 0:
            # Positive → label to the RIGHT of the bar
            plt.text(
                width + 0.01 * x_max,
                y,
                f"{val:.3f}\nws = {ws:.0f}",
                va="center",
                ha="left",
                fontsize=10,
                color="black"
            )
        else:
            # Negative → label to the LEFT of the bar
            plt.text(
                width - 0.01 * abs(x_min),
                y,
                f"{val:.3f}\nws = {ws:.0f}",
                va="center",
                ha="right",
                fontsize=10,
                color="black"
            )

    plt.title(title, fontsize=16)
    plt.xlabel("Importance Score")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    # plt.show()