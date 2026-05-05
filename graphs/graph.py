import numpy as np
import matplotlib.pyplot as plt

def annotate_bars(bars, fmt="{:.2f}", offset=3):
    """
    Annotate bar values on top of bars.
    - bars: container returned by plt.bar
    - fmt: number format
    - offset: vertical offset in points
    """
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )

# =========================
# Data (EDIT THIS SECTION)
# =========================
schemes = [
    "LLM-Only",
    "RAG",
    "HRAG",
    "HRAG-FT",
    "HRAG-FT-R"
]

# Values for the two valuation schemes
valuation_1 = [2.37, 4.07, 4.18, 5.06, 4.76]
valuation_2 = [2.71, 4.60, 5.11, 5.77, 5.22]
valuation_3 = [4.38, 4.85, 6.64, 6.90, 5.50]

# Optional error bars (set to None if not needed)
err_1 = [0.02, 0.015, 0.018, 0.017, 0.016]
err_2 = [0.021, 0.017, 0.019, 0.016, 0.018]
err_3 = [0.021, 0.017, 0.019, 0.016, 0.018]

# =========================
# Plot configuration
# =========================
x = np.arange(len(schemes))
bar_width = 0.30

plt.figure(figsize=(7, 4))

# Bars
bars1 = plt.bar(
    x - bar_width,
    valuation_1,
    bar_width,
    #yerr=err_1,
    capsize=4,
    label="LLM Judges: NetConfRAG-17",
    hatch="///"
)

bars2 = plt.bar(
    x,
    valuation_2,
    bar_width,
    #yerr=err_2,
    capsize=4,
    label="Human Expert: NetConfRAG-17",
    hatch="\\\\\\"
)

bars3 = plt.bar(
    x + bar_width,
    valuation_3,
    bar_width,
    #yerr=err_2,
    capsize=4,
    label="LLM Judges: NetConfRAG-144",
    hatch="//"
)
annotate_bars(bars1)
annotate_bars(bars2)
annotate_bars(bars3)

# =========================
# Axes and labels
# =========================
plt.ylabel("Performance Score", fontsize=11)
plt.xlabel("Configuration generation approach", fontsize=11)
plt.xticks(x, schemes, fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 10)

plt.legend(
    frameon=False,
    fontsize=10,
    loc="upper left"
)

# Grid (subtle, paper-friendly)
plt.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.6
)

# Tight layout for LaTeX / paper inclusion
plt.tight_layout()


# =========================
# Export (RECOMMENDED)
# =========================
plt.savefig("comparison_bar_chart.pdf", dpi=300, bbox_inches="tight")
plt.savefig("comparison_bar_chart.png", dpi=300, bbox_inches="tight")

plt.show()

