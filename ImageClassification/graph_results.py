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
    df = pd.read_sql_query("SELECT id, model_name, num_params, test_accuracy, val_accuracy, total_wattage, avg_wattage FROM results", conn)
    conn.close()

    # Set variable names to selected values from DB
    id = df['id']
    model_name = df['model_name']
    num_params = df['num_params']
    test_acc = df['test_accuracy']
    val_acc = df['val_accuracy']
    total_wattage = df['total_wattage']
    avg_wattage = df['avg_wattage']

    # Scatter plot of param count vs. accuracy vs. total wattage in all combinations
    scatter_plot_2d(num_params, val_acc, "Param Count", "Accuracy (%)", 1)
    scatter_plot_2d(val_acc, total_wattage, "Accuracy(%)", "Total Wattage", 2)
    scatter_plot_2d(total_wattage, num_params, "Total Wattage", "Param Count", 3)

    # conn = sqlite3.connect('3cnnlayer_3fclayer_128_results.db')
    # cursor = conn.cursor()
    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # print(cursor.fetchall())
    # df = pd.read_sql_query("SELECT * FROM results", conn)
    # conn.close()
    # epoch = df['epoch']
    # test_acc = df['test_accuracy']
    # power_draw = df['total_wattage']
    # param_count = df['num_params']
    # scatter_plot_2d(epoch, test_acc, "Epoch", "Test Accuracy", 1)
    # scatter_plot_2d(test_acc, power_draw, "Test Accuracy", "GPU Wattage", 2)


if __name__=="__main__":
    main()