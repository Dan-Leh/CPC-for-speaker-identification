import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_results():
    data_percentages = [1, 2, 5, 10, 20, 50, 100]
    methods = ['Fully_supervised']
    max_vals = np.zeros((len(methods),len(data_percentages)))
    save_dir = "Results"
    for j, method in enumerate(methods):
        for i, perc in enumerate(data_percentages):
            with open(f"trained_models/{method}_results/{method}_{perc}/val_metrics.csv", "r") as f:
                lines = f.readlines()
                f.close()
            lines = [line.split(",") for line in lines]
            vals = [float(line[2]) for line in lines[1:]]
            max_vals[j,i] = max(vals)/100
        plt.plot(data_percentages, max_vals[j,:], label=method, marker='o')
    
    plt.legend()
    plt.grid(True, 'both', 'both')
    plt.xscale('log')
    plt.xticks(data_percentages, [str(perc) for perc in data_percentages])
    plt.xlabel("Percentage of labeled data")
    plt.ylabel("Classificaton accuracy")
    plt.ylim(0, 1)
    
    plt.savefig(os.path.join(save_dir, 'results.png'))
    return 
    

if __name__ == "__main__":
    visualize_results()

