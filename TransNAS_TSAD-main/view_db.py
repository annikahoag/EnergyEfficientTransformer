import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def view_results():
    # Connect database
    conn = sqlite3.connect("results.db")
    c = conn.cursor()

    # Fetch all rows from the study_reuslts table
    c.execute("SELECT * FROM study_results")
    rows = c.fetchall()

    # Print each row with formatting 
    for row in rows:
        # trial_number = row[0]
        F1_value = row[1]
        params = row[2]
        #params = json.loads(params_json) #turn JSON string back into dict
        GPU_value = row[4]
        fitness = row[5]

        print(f"F1 value={F1_value}, params={params}, GPU usage={GPU_value}, Fitness={fitness}")
    
    conn.close()


#LEFT OFF: GRAPH THE RESULTS FROM 10 GENERATIONS

def graph_results_2d():
    conn = sqlite3.connect("results.db")
    df = pd.read_sql_query("SELECT trial_number, F1_value, params, GPU_usage FROM study_results", conn)
    
    conn.close()

    # df_high = df[df['F1_value'] > 0.9]
    plt.figure(num=1)
    plt.scatter(df['params'], df['GPU_usage'], c='blue', marker='o')
    plt.xlabel("Param Count")
    plt.ylabel("GPU Usage")
    plt.grid(True)
    #plt.show()
    plt.savefig("Param Count vs GPU Usage")

    plt.figure(num=2)
    plt.scatter(df['params'], df['F1_value'], c='blue', marker='o')
    plt.xlabel("Param Count")
    plt.ylabel("F1 Score")
    plt.grid(True)
    #plt.show()
    plt.savefig("Param Count vs F1 score")

    plt.figure(num=3)
    plt.scatter(df['GPU_usage'], df['F1_value'], c='blue', marker='o')
    plt.xlabel("GPU Usage")
    plt.ylabel("F1 Score")
    plt.grid(True)
    #plt.show()
    plt.savefig("GPU Usage vs F1 score")
    # # plt.pause(0.001)
    # plt.figure(figsize=(6,4))
    # plt.scatter(
    #     df_high['params'],
    #     df_high['GPU_usage'],
    #     c='C0',
    #     marker='o',
    #     edgecolor='k',
    #     alpha=0.8,
    #     label='F1 > 0.9'
    # )
    # plt.xlabel("Param Count")
    # plt.ylabel("GPU Usage")
    # plt.title("Param Count vs GPU Usage (F1 > 0.9)")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("param_gpu.png")

    


def graph_results_scatter():
    conn = sqlite3.connect("results.db")
    df = pd.read_sql_query("SELECT trial_number, F1_value, params, GPU_usage FROM study_results", conn)

    conn.close()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['F1_value'], df['params'], df['GPU_usage'], c='blue', marker='o')

    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('GPU Usage')
    ax.set_zlabel('F1 Score')

    
    ax.set_title('3D Scatter Plot: F1 Score vs Parameters vs GPU Usage')

    plt.tight_layout()

    plt.savefig("pareto_front.png")


def graph_results_curve():
   
    conn = sqlite3.connect("results.db")
    df = pd.read_sql_query(
        "SELECT trial_number, F1_value, params, GPU_usage FROM study_results",
        conn
    )
    conn.close()

    # 2. Prepare X, y
    X = df[['params', 'GPU_usage']].values
    y = df['F1_value'].values

    # 3. Build a polynomial regression model (e.g. degree=2 or 3)
    degree = 2
    model = make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X, y)

    # 4. Create a grid for plotting the fitted surface
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # 5. Predict on the grid and reshape
    zz = model.predict(grid_points).reshape(xx.shape)

    # 6. Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # scatter original data
    ax.scatter(X[:, 0], X[:, 1], y,
            color='blue', alpha=0.6, label='Data points')

    # plot polynomial surface
    ax.plot_surface(xx, yy, zz,
                    color='orange', alpha=0.5, rstride=1, cstride=1,
                    antialiased=True)

    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('GPU Usage')
    ax.set_zlabel('F1 Score')
    ax.set_title(f'3D Polynomial Fit (degree={degree})')
    ax.legend()

    plt.tight_layout()
    plt.show()
    # plt.savefig("pareto_front.png")



def graph_residuals():
    
    conn = sqlite3.connect("results.db")
    df = pd.read_sql_query("SELECT trial_number, F1_value, params, GPU_usage FROM study_results", conn)

    conn.close()

    X = df[['params', 'GPU_usage']].values
    y = df['F1_value'].values

    model = LinearRegression()
    model.fit(X, y) 
    y_pred = model.fit(X, y).predict(X)
    
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    residuals = y - y_pred

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Data points')

    ax.plot_surface(xx, yy, zz, color='orange', alpha=0.5, label='Fitted surface')

    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('GPU Usage')
    ax.set_zlabel('F1 Score')
    ax.set_title('3D Fit: F1 Score vs Parameters vs GPU Usage')
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Predicted Param Count')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Residual Plot for 3D Regression Model')
    plt.grid(True)


def graph_cmaes():
    # Load database
    conn = sqlite3.connect("results.db")
    df = pd.read_sql_query(
        "SELECT id, F1_value, params, GPU_usage, GPU_avg, fitness FROM study_results",
        conn
    )
    conn.close()

    # x = df["F1_value"].values
    # y = df["GPU_avg"].values
    # z = df["params"].values
    x = df["params"].values
    y = df["GPU_avg"].values


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
    sc = plt.scatter(x, y)
    # plt.colorbar(sc, label="Parameter Count")


    # Labels
    plt.xlabel("Parameter Count")
    plt.ylabel("Avg Wattage")
    # plt.title("Pareto Front for F1 Score and GPU Wattage")
    # plt.legend()
    plt.tight_layout()
    plt.show()

        

view_results()
# graph_cmaes()
graph_results_2d()
# graph_results_curve()