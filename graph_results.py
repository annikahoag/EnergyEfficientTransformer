import sqlite3
import pandas as pd 
import matplotlib.pyplot as plt



# Graph scatter plot for two values
def scatter_plot_2d(xvals, yvals, xtitle, ytitle, fig_num):
    print("in function")
    plt.figure(num=fig_num)
    plt.scatter(xvals, yvals, c='blue', marker='o')
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.grid(True)
    plt.show()



def main():
    conn = sqlite3.connect('final_results.db')
    df = pd.read_sql_query("SELECT id, model_name, num_params, accuracy, total_wattage, avg_wattage FROM results", conn)
    conn.close()

    # Set variable names to selected values from DB
    id = df['id']
    model_name = df['model_name']
    num_params = df['num_params']
    accuracy = df['accuracy']
    total_wattage = df['total_wattage']
    avg_wattage = df['avg_wattage']

    # Scatter plot of param count vs. accuracy vs. total wattage in all combinations
    scatter_plot_2d(num_params, accuracy, "Param Count", "Accuracy (%)", 1)
    scatter_plot_2d(accuracy, total_wattage, "Accuracy(%)", "Total Wattage", 2)
    scatter_plot_2d(total_wattage, num_params, "Total Wattage", "Param Count", 3)



if __name__=="__main__":
    main()