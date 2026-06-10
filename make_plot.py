import glob
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["cora", "camera", "funding", "voters", "cddb", "restaurant", "wdc20", "wdc50", "wdc80", "census", "synth_250", "synth_1000", "synth_5000", "synth_10000"])
    parser.add_argument('--batch_size', type=int)
       
    args = parser.parse_args()


    dname = args.dataset
    batch_size = args.batch_size


with open('results/Phi.json', 'r') as f:
    dict_min = json.load(f)


phi = dict_min[dname+"_"+"Phi"][str(batch_size)][0]
max_query = phi * 3


if dname != "synth_10000":
    file_pattern = "results/"+dname+"/"+dname+"_*.csv"

    plt.figure()
    dfs = {}

    plt.figure(figsize=(4, 2))

    results = pd.read_csv("results/"+dname+"/"+dname+"_suboptimal"+","+str(batch_size)+".csv")
    list_result = results["recall"].values.tolist()[:max_query]
    if batch_size == 2:
        plt.plot(range(0,len(list_result)+1), [0]+list_result, alpha=0.8, linestyle=':', label="Opt", color = "green")
    else:
        plt.plot(range(0,len(list_result)+1), [0]+list_result, alpha=0.8, linestyle=':', label="SubOpt", color = "green")
    print("SUBOPTIMAL AT PHI:", round(list_result[int(max_query/3)-1],6) )

    results = pd.read_csv("results/"+dname+"/"+dname+"_pERbacco"+","+str(batch_size)+",lou,0.05.csv")
    list_result = results["recall"].values.tolist()[:max_query]
    plt.plot(range(0,len(list_result)+1), [0]+list_result, alpha=0.8, linestyle='-', label="pERbacco", color = "red")

    results = pd.read_csv("results/"+dname+"/"+dname+"_pERbac,"+str(batch_size)+".csv")
    list_result = results["recall"].values.tolist()[:max_query]
    plt.plot(range(0,len(list_result)+1), [0]+list_result, alpha=0.8, linestyle='--', label="pERbac", color = "blue")


    results = pd.read_csv("results/"+dname+"/"+dname+"_Online,"+str(batch_size)+".csv")
    list_result = results["recall"].values.tolist()[:max_query]
    plt.plot(range(0,len(list_result)+1), [0]+list_result, alpha=0.8, linestyle='-.', label="Online", color = "orange")


    plt.ylim(0,1.05)

    if dname == "voters":
        plt.legend(loc='upper right', fontsize=11)
    else:
        plt.legend(loc='lower right', fontsize=11)
    plt.grid(True)

    plt.xlabel(f"number query ($\\phi_{{{batch_size}}} = {phi}$)", fontsize=11)

    plt.yticks(rotation=45)
    plt.tick_params(axis='y', labelsize=8)
    plt.ylabel("recall", fontsize=11)


    x_min, x_max = plt.xlim()  # get current x-axis limits
    ticks = np.linspace(0, max_query, 4)

    tick_labels = [0, f"$\\phi_{{{batch_size}}}$", f"$2\\phi_{{{batch_size}}}$", f"$3\\phi_{{{batch_size}}}$",]
    plt.margins(0)
    plt.subplots_adjust(left=0.22, right=0.88)

    plt.xticks(ticks, tick_labels)
    plt.xticks(ticks)

    plt.savefig("PLOT/"+dname+"_"+str(batch_size)+".pdf", bbox_inches='tight',pad_inches=0)



else:
    for synth_precision in ["0.5", "0.2", "0.05"]:
        
        max_query = dict_min[dname+"_"+"Phi"][str(batch_size)][0] * 3
        phi = dict_min[dname+"_"+"Phi"][str(batch_size)][0]
        print(max_query)

        recall_blocking = dict_min[dname]["recall"]
        file_pattern = "results/"+dname+"/"+dname+"_*.csv"

        plt.figure()
        dfs = {}

        plt.figure(figsize=(4, 2))


        results = pd.read_csv("results/"+dname+"/"+dname+"_suboptimal"+","+str(batch_size)+".csv")
        list_result = results["recall"].values.tolist()[:max_query]
        if batch_size == 2:
            plt.plot(range(0,len(list_result)+1), [0]+list_result, alpha=0.8, linestyle=':', label="Opt", color = "green")
        else:
            plt.plot(range(0,len(list_result)+1), [0]+list_result, alpha=0.8, linestyle=':', label="SubOpt", color = "green")
        print("SUBOPTIMAL AT Phi:", round(list_result[int(max_query/3)-1],16))


        results = pd.read_csv("results/"+dname+"/"+dname+","+str(synth_precision)+","+str(batch_size)+",lou,brmean.csv")
        list_result = results["recall"].values.tolist()[:max_query]
        plt.plot(range(0,len(list_result)+1), [0]+list_result, alpha=0.8, linestyle='-', label="pERbacco", color = "red")

        results = pd.read_csv("results/"+dname+"/"+dname+","+str(synth_precision)+","+str(batch_size)+",Fal,brmean.csv")
        list_result = results["recall"].values.tolist()[:max_query]
        plt.plot(range(0,len(list_result)+1), [0]+list_result, alpha=0.8, linestyle='--', label="pERbac", color = "blue")

        
        results = pd.read_csv("results/"+dname+"/"+dname+","+str(synth_precision)+","+str(batch_size)+",Fal,brmax.csv")
        list_result = results["recall"].values.tolist()[:max_query]
        plt.plot(range(0,len(list_result)+1), [0]+list_result, alpha=0.8, linestyle='-.', label="Online", color = "orange")

        plt.ylim(0,1.05)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True)

        if batch_size == 10:
            plt.xlabel("number query " + r"($\phi_{10} = $" + str(phi)+")", fontsize=11)
        if batch_size == 2:
            plt.xlabel("number query " + r"($\phi_{2} = $" + str(phi)+")", fontsize=11)

        plt.yticks(rotation=45)
        plt.tick_params(axis='y', labelsize=8)
        plt.ylabel("recall", fontsize=11)


        x_min, x_max = plt.xlim()  # get current x-axis limits

        ticks = np.linspace(0, max_query, 4)
        if batch_size == 10:
            tick_labels = [0, r"$\phi_{10}$", r"$2\phi_{10}$", r"$3\phi_{10}$"]
        if batch_size == 2:
            tick_labels = [0, r"$\phi_{2}$", r"$2\phi_{2}$", r"$3\phi_{2}$"]

        plt.margins(0)

        plt.subplots_adjust(left=0.22, right=0.88)

        plt.xticks(ticks, tick_labels)
        plt.xticks(ticks)

        plt.savefig("PLOT/"+dname+"_"+str(batch_size)+"_"+str(synth_precision)+".pdf", bbox_inches='tight',pad_inches=0)

