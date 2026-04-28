import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# ML-MATT-CompetitionQT2021 Experiment Results
# ============================================================

epochs = np.arange(1, 31)

# ------------------------------------------------------------
# Transformer only / MFE only
# ------------------------------------------------------------
mfe_valid_mae = [
    0.5180, 0.5186, 0.4858, 0.5268, 0.5753,
    0.5056, 0.4846, 0.4691, 0.4697, 0.4539,
    0.4360, 0.4512, 0.4590, 0.5034, 0.5218,
    0.4588, 0.5361, 0.4770, 0.4719, 0.5267,
    0.5345, 0.5006, 0.4974, 0.4871, 0.5508,
    0.4943, 0.4821, 0.5707, 0.5229, 0.5079
]

mfe_valid_rmse = [
    0.7606, 0.7497, 0.7322, 0.7380, 0.8019,
    0.7472, 0.7334, 0.7345, 0.7280, 0.7268,
    0.7280, 0.7461, 0.7384, 0.7647, 0.7626,
    0.7414, 0.8018, 0.7629, 0.7721, 0.8059,
    0.7895, 0.7835, 0.7742, 0.8014, 0.8243,
    0.7870, 0.7855, 0.8839, 0.8249, 0.8115
]

# ------------------------------------------------------------
# TCN only / TCF only
# ------------------------------------------------------------
tcf_valid_mae = [
    0.8502, 0.5394, 0.4818, 0.4810, 0.4945,
    0.5791, 0.4781, 0.4489, 0.4662, 0.4419,
    0.4827, 0.4986, 0.4575, 0.4500, 0.4415,
    0.4877, 0.4621, 0.4486, 0.5500, 0.4876,
    0.4568, 0.4599, 0.4616, 0.4456, 0.4507,
    0.4903, 0.4654, 0.4611, 0.4703, 0.4879
]

tcf_valid_rmse = [
    1.0014, 0.7686, 0.7306, 0.7305, 0.7255,
    0.7683, 0.7181, 0.7282, 0.7304, 0.7060,
    0.7199, 0.7219, 0.7168, 0.7089, 0.7225,
    0.7204, 0.7221, 0.7240, 0.7709, 0.7463,
    0.7518, 0.7294, 0.7404, 0.7425, 0.7443,
    0.7557, 0.7754, 0.7351, 0.7518, 0.7521
]

# ------------------------------------------------------------
# Transformer + TCN / Hybrid
# ------------------------------------------------------------
hybrid_valid_mae = [
    0.5969, 0.5255, 0.5479, 0.5410, 0.4722,
    0.4616, 0.4399, 0.4487, 0.4580, 0.4600,
    0.4519, 0.4411, 0.4757, 0.4685, 0.5112,
    0.4626, 0.5292, 0.5160, 0.5364, 0.5355,
    0.4644, 0.4675, 0.4853, 0.5566, 0.5468,
    0.4776, 0.4786, 0.5399, 0.5150, 0.5163
]

hybrid_valid_rmse = [
    0.8362, 0.7811, 0.7731, 0.7593, 0.7147,
    0.7014, 0.7061, 0.7331, 0.7047, 0.7487,
    0.7118, 0.7195, 0.7237, 0.7384, 0.7437,
    0.7235, 0.8033, 0.7406, 0.7696, 0.7605,
    0.7549, 0.7542, 0.7487, 0.7844, 0.7863,
    0.7487, 0.7693, 0.7771, 0.7751, 0.7583
]

# ------------------------------------------------------------
# Final Test Result
# ------------------------------------------------------------
model_names = ["MFE only", "TCF only", "MFE+TCF"]
final_mae = [0.1882, 0.2082, 0.2252]
final_rmse = [0.2943, 0.2780, 0.2920]

# ============================================================
# Plot
# ============================================================
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 12

fig, axs = plt.subplots(2, 2)
fig.suptitle(
    "Experimental Results on ML-MATT-CompetitionQT2021 Dataset",
    fontsize=20,
    fontweight="bold"
)

# ------------------------------------------------------------
# 1. Validation MAE Curve
# ------------------------------------------------------------
axs[0, 0].plot(epochs, mfe_valid_mae, marker="o", linewidth=2, label="MFE only")
axs[0, 0].plot(epochs, tcf_valid_mae, marker="o", linewidth=2, label="TCF only")
axs[0, 0].plot(epochs, hybrid_valid_mae, marker="o", linewidth=2, label="MFE+TCF")

axs[0, 0].set_title("Validation MAE Curve")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Valid MAE")
axs[0, 0].grid(True, linestyle="--", alpha=0.4)
axs[0, 0].legend()

# ------------------------------------------------------------
# 2. Validation RMSE Curve
# ------------------------------------------------------------
axs[0, 1].plot(epochs, mfe_valid_rmse, marker="o", linewidth=2, label="MFE only")
axs[0, 1].plot(epochs, tcf_valid_rmse, marker="o", linewidth=2, label="TCF only")
axs[0, 1].plot(epochs, hybrid_valid_rmse, marker="o", linewidth=2, label="MFE+TCF")

axs[0, 1].set_title("Validation RMSE Curve")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Valid RMSE")
axs[0, 1].grid(True, linestyle="--", alpha=0.4)
axs[0, 1].legend()

# ------------------------------------------------------------
# 3. Final Test MAE Bar
# ------------------------------------------------------------
x = np.arange(len(model_names))

bars1 = axs[1, 0].bar(x, final_mae)
axs[1, 0].set_title("Final Test MAE")
axs[1, 0].set_ylabel("MAE")
axs[1, 0].set_xticks(x)
axs[1, 0].set_xticklabels(model_names)
axs[1, 0].grid(True, axis="y", linestyle="--", alpha=0.4)

for bar in bars1:
    h = bar.get_height()
    axs[1, 0].text(
        bar.get_x() + bar.get_width() / 2,
        h,
        f"{h:.4f}",
        ha="center",
        va="bottom"
    )

# ------------------------------------------------------------
# 4. Final Test RMSE Bar
# ------------------------------------------------------------
bars2 = axs[1, 1].bar(x, final_rmse)
axs[1, 1].set_title("Final Test RMSE")
axs[1, 1].set_ylabel("RMSE")
axs[1, 1].set_xticks(x)
axs[1, 1].set_xticklabels(model_names)
axs[1, 1].grid(True, axis="y", linestyle="--", alpha=0.4)

for bar in bars2:
    h = bar.get_height()
    axs[1, 1].text(
        bar.get_x() + bar.get_width() / 2,
        h,
        f"{h:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.subplots_adjust(top=0.90)

plt.savefig("ml_matt_experiment_results.png", dpi=300, bbox_inches="tight")
plt.show()