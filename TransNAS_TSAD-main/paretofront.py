import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

# # --- Load Data ---
# conn = sqlite3.connect("results.db")
# df = pd.read_sql_query("SELECT * FROM study_results", conn)
# df = df[~(df == 0).any(axis=1)]

# # --- Pareto Filtering ---
# def pareto_mask(costs: np.ndarray) -> np.ndarray:
#     is_eff = np.ones(costs.shape[0], dtype=bool)
#     for i, c in enumerate(costs):
#         if is_eff[i]:
#             is_eff[is_eff] = np.any(costs[is_eff] > c, axis=1) | np.all(costs[is_eff] == c, axis=1)
#             is_eff[i] = True
#     return is_eff

# # costs = np.column_stack((-df["F1_value"], df["params"], df["GPU_usage"]))
# # mask = pareto_mask(costs)

# # pareto_df = (
# #     df.loc[mask, ["F1_value", "params", "GPU_usage"]]
# #     .sort_values("F1_value", ascending=False)
# #     .reset_index(drop=True)
# # )

# # # --- Polynomial Fit (3D parametric curve) ---
# # t = np.linspace(0, 1, len(pareto_df))  # parametric variable

# # # Fit polynomials of degree 3
# # deg = 3
# # px = np.polyfit(t, pareto_df["F1_value"], deg)
# # py = np.polyfit(t, pareto_df["params"], deg)
# # pz = np.polyfit(t, pareto_df["GPU_usage"], deg)

# # # # Evaluate fit
# # # t_fit = np.linspace(0, 1, 200)
# # x_fit = np.polyval(px, t_fit)
# # y_fit = np.polyval(py, t_fit)
# # z_fit = np.polyval(pz, t_fit)

# # # --- 3D Plot ---
# # fig = plt.figure(figsize=(11, 9))
# # ax = fig.add_subplot(111, projection="3d")
# # ax.scatter(df["F1_value"], df["params"], df["GPU_usage"], alpha=0.25, label="All Trials")
# # ax.scatter(pareto_df["F1_value"], pareto_df["params"], pareto_df["GPU_usage"],
# #            color="red", label="Pareto Points")
# # ax.plot(x_fit, y_fit, z_fit, color="black", linewidth=2.5, label="Polynomial Fit")

# # ax.set_xlabel("F1 Score (↑)")
# # ax.set_ylabel("Parameter Count (↓)")
# # ax.set_zlabel("GPU Usage (↓)")
# # ax.set_title("3D Polynomial Fit to Pareto Front")
# # ax.legend()
# # plt.tight_layout()
# # plt.show()

# # --- 2D Projections ---
# fig, axs = plt.subplots(1, 3, figsize=(18, 5))
# axs[0].scatter(df["params"], df["F1_value"], alpha=0.25)
# axs[0].plot(y_fit, x_fit, color="red")
# axs[0].set_xlabel("Parameter Count")
# axs[0].set_ylabel("F1 Score")
# axs[0].set_title("F1 vs Parameters")

# axs[1].scatter(df["GPU_usage"], df["F1_value"], alpha=0.25)
# axs[1].plot(z_fit, x_fit, color="red")
# axs[1].set_xlabel("GPU Usage")
# axs[1].set_ylabel("F1 Score")
# axs[1].set_title("F1 vs GPU Usage")

# axs[2].scatter(df["GPU_usage"], df["params"], alpha=0.25)
# axs[2].plot(z_fit, y_fit, color="red")
# axs[2].set_xlabel("GPU Usage")
# axs[2].set_ylabel("Parameter Count")
# axs[2].set_title("Params vs GPU Usage")

# plt.tight_layout()
# plt.show()


def plot_2_objectives():

    # Load database
    conn = sqlite3.connect("results.db")
    df = pd.read_sql_query(
        "SELECT trial_number, F1_value, params, GPU_usage, GPU_avg FROM study_results",
        conn
    )
    conn.close()

    x = df["F1_value"].values
    y = df["GPU_avg"].values
    z = df["params"].values


    # Fit curve
    coeffs = np.polyfit(x, y, deg=1)
    poly  = np.poly1d(coeffs)

    # Generate smooth x-values for the curve
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = poly(x_fit)


    
    # Plot curve with low opacity so we can see the color of the points
    plt.figure(figsize=(8,5))
    plt.plot(x_fit, y_fit,
         color="red",
         lw=2,
         alpha=0.4,      # make the curve lighter
         zorder=1,       # draw it underneath the points
         label="2nd-degree fit")


    # Make scatter plot of F1 and wattage, color gradient based on parameter count value
    sc = plt.scatter(x, y, c=z, cmap="viridis", edgecolor="k", alpha=0.8, zorder=2)
    plt.colorbar(sc, label="Parameter Count")


    # Labels
    plt.xlabel("F1 Score")
    plt.ylabel("GPU Wattage")
    plt.title("Pareto Front for F1 Score and GPU Wattage")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_2_objectives()