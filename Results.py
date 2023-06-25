import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_results():
    data_percentages = [1, 2, 5, 10, 20, 50, 100]
    methods = ['CLE','CUE','FS']
    max_vals = np.zeros((len(methods),len(data_percentages)))
    
    for j, method in enumerate(methods):
        for i, perc in enumerate(data_percentages):
            with open(f"Results/{method}_results/{method}_results_{perc}/val_metrics.csv", "r") as f:
                lines = f.readlines()
                f.close()
            lines = [line.split(",") for line in lines]
            vals = [float(line[2]) for line in lines[1:]]
            max_vals[j,i] = max(vals)/100
        plt.plot(data_percentages, max_vals[j,:], label=methods[j], marker='o')
    
    plt.legend()
    plt.grid(True, 'both', 'both')
    plt.xscale('log')
    plt.xticks(data_percentages, [str(perc) for perc in data_percentages])
    plt.xlabel("Percentage of labeled data")
    plt.ylabel("Classificaton accuracy")
    plt.ylim(0, 1)
    
    plt.savefig(os.path.join('Results', 'Results.png'))
    print("Max values:")
    print(max_vals)
    return 
    

if __name__ == "__main__":
    visualize_results()

