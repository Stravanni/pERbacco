from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["all", "cora", "camera", "funding", "voters","wdc80", "synth_10000"])

    args = parser.parse_args()
    dataset = args.dataset

if dataset == "all":
    datasets = ["cora", "camera", "funding", "voters","wdc80", "synth_10000"]
else:
    datasets = [dataset]


def output_path_for(dataset, batch_size, alg_community, lambda_w, mu_benefit, optimal, synth_precision):
    directory = Path("results") / dataset
    if optimal == "True":
        return directory / f"{dataset}_suboptimal,{batch_size}.csv"

    if "synth" in dataset:
        return directory / f"{dataset},{synth_precision},{batch_size},{alg_community[:3]},{mu_benefit}.csv"

    if alg_community != "False":
        return directory / f"{dataset}_pERbacco,{batch_size},{alg_community[:3]},{lambda_w}.csv"

    if mu_benefit == "brmean":
        return directory / f"{dataset}_pERbac,{batch_size}.csv"

    return directory / f"{dataset}_Online,{batch_size}.csv"

# Loop over each dataset and run the script
for dataset in datasets:
    #for batch_size in ["40", "20", "5"]:
    for batch_size in ["10"]:
            
        # ONLINE/PERBAC
        if True: 
            for mu_benefit in ["brmean","brmax"]:
                if "synth" in dataset:
                    list_synth_precision = ["1","0.5","0.2","0.05"]
                else:
                    list_synth_precision = ["False"]
                for synth_precision in list_synth_precision:
                    output_path = output_path_for(dataset, batch_size, "False", "False", mu_benefit, "False", synth_precision)
                    if output_path.exists():
                        print(f"Skipping existing output: {output_path}", flush=True)
                        continue
                    command = [sys.executable, "perbacco.py", "--dataset", dataset, "--batch_size", batch_size,
                               "--alg_community", "False", "--lambda_w", "False", "--mu_benefit", mu_benefit,  "--optimal", "False",
                               "--synth_precision", synth_precision]
                    print("Running:", " ".join(command), flush=True)
                    subprocess.run(command, check=True)
        
            
            # PERBACCO
            if True: 
                for alg_community in ["louvain"]:                    
                    for lambda_w in ["0.05"]:
                        if "synth" in dataset:
                            list_synth_precision = ["1","0.5","0.2","0.05"]
                        else:
                            list_synth_precision = ["False"]
                        for synth_precision in list_synth_precision:
                            for mu_benefit in ["brmean"]:
                                output_path = output_path_for(dataset, batch_size, alg_community, lambda_w, mu_benefit, "False", synth_precision)
                                if output_path.exists():
                                    print(f"Skipping existing output: {output_path}", flush=True)
                                    continue
                                command = [sys.executable, "perbacco.py", "--dataset", dataset, "--batch_size", batch_size,
                                           "--alg_community", alg_community, "--lambda_w", lambda_w, "--mu_benefit", mu_benefit, "--optimal", "False",
                                           "--synth_precision", synth_precision]
                                print("Running:", " ".join(command), flush=True)
                                subprocess.run(command, check=True)
        
        # OPTIMAL SOLUTION
        if True:
            if "synth" in dataset:
                list_synth_precision = ["1"] # DO NOT CHANGE!!!!!!
            else:
                list_synth_precision = ["False"]
            for synth_precision in list_synth_precision:
                output_path = output_path_for(dataset, batch_size, "False", "False", "brmax", "True", synth_precision)
                if output_path.exists():
                    print(f"Skipping existing output: {output_path}", flush=True)
                    continue
                command = [sys.executable, "perbacco.py", "--dataset", dataset, "--batch_size", batch_size,
                           "--alg_community", "False", "--lambda_w", "False", "--mu_benefit", "brmax", "--optimal", "True",
                           "--synth_precision", synth_precision]
                print("Running:", " ".join(command), flush=True)
                subprocess.run(command, check=True)

