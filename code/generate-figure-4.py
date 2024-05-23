import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    ndcg_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)


def rank_scores(y_true, y_score):
    score = ndcg_score(np.asarray([y_true]), np.asarray([y_score]))
    coeff = spearmanr(y_true, y_score)
    print(f"Between {y_true} and {y_score}:")
    print(f"nDCG score is {score}")
    print(f"Spearman rank coefficient is {coeff.statistic}")
    return score, coeff


if __name__ == "__main__":
    root_path = "/Users/amithkamath/data/"
    expert_path = os.path.join(root_path, "adoser-data/expert/")
    data_path = os.path.join(root_path, "adoser-data/experiments/")
    reference_path = os.path.join(data_path, "reference")
    prediction_path = os.path.join(data_path, "predicted")

    jw_results = pd.read_csv(os.path.join(expert_path, "jw_impact.csv"))
    jw_results = jw_results.iloc[:, [1, 2, 3, 4]].stack().reset_index()

    ee_results = pd.read_csv(os.path.join(expert_path, "ee_impact.csv"))
    ee_results = ee_results.iloc[:, [1, 2, 3, 4]].stack().reset_index()

    er_results = pd.read_csv(os.path.join(expert_path, "er_impact.csv"))
    er_results = er_results.iloc[:, [1, 2, 3, 4]].stack().reset_index()

    type = "max"
    n_oar_gt = 1
    thresh_gt = 0.1
    output_path = os.path.join(
        os.path.join(root_path, "adoser-data/results/"),
        str(thresh_gt) + "_" + str(n_oar_gt) + "_" + type,
    )
    os.makedirs(output_path, exist_ok=True)

    cases = [70, 71, 72, 73, 74, 75, 76, 78, 80, 81, 82, 83, 84, 85, 86]
    list_oar_names = [
        "BrainStem",
        "Chiasm",
        "Cochlea_L",
        "Cochlea_R",
        "Eye_L",
        "Eye_R",
        "Hippocampus_L",
        "Hippocampus_R",
        "LacrimalGland_L",
        "LacrimalGland_R",
        "OpticNerve_L",
        "OpticNerve_R",
        "Pituitary",
    ]

    results = []
    for case in cases:
        out = []
        if case == 75:
            out = [None, None, None, None]
        else:
            data = pd.read_csv(
                os.path.join(
                    reference_path,
                    "ISAS_GBM_" + str(case).zfill(3) + "_" + type + ".csv",
                ),
                index_col=0,
            )

            percentage_diff = (
                data[["1", "2", "3", "4"]].div(data["0"], axis=0).loc[list_oar_names]
            )

            for idx in range(1, 5):
                if np.any(
                    np.sum(percentage_diff[[str(idx)]] > thresh_gt, axis=0) >= n_oar_gt
                ):
                    out.append("Worse")
                elif np.any(
                    np.sum(percentage_diff[[str(idx)]] < -thresh_gt, axis=0) >= n_oar_gt
                ):
                    out.append("Better")
                else:
                    out.append("No Change")
        out.insert(0, str(case))
        results.append(out)
    reference_results = pd.DataFrame(results)
    reference_results.iloc[7, 4] = None  # Handle case 78
    reference_results.iloc[10, 4] = None  # Handle case 82
    reference_results.to_csv(os.path.join(output_path, "reference_results.csv"))
    reference_results = reference_results.loc[:, 1:].stack().reset_index()

    precision_map = {"Pred": {}, "R1": {}, "R2": {}, "R3": {}}
    recall_map = {"Pred": {}, "R1": {}, "R2": {}, "R3": {}}
    f1_map = {"Pred": {}, "R1": {}, "R2": {}, "R3": {}}
    for alpha in [
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
    ]:
        for n_oar_pred in range(9, 0, -1):
            thresh_pred = alpha * thresh_gt
            results = []
            for case in cases:
                out = []
                if case == 75:
                    out = [None, None, None, None]
                else:
                    data = pd.read_csv(
                        os.path.join(
                            prediction_path,
                            "ISAS_GBM_" + str(case).zfill(3) + "_" + type + ".csv",
                        ),
                        index_col=0,
                    )

                    percentage_diff = (
                        data[["1", "2", "3", "4"]]
                        .div(data["0"], axis=0)
                        .loc[list_oar_names]
                    )

                    for idx in range(1, 5):
                        if np.any(
                            np.sum(percentage_diff[[str(idx)]] > thresh_pred, axis=0,)
                            >= n_oar_pred
                        ):
                            out.append("Worse")
                        elif np.any(
                            np.sum(percentage_diff[[str(idx)]] < -thresh_pred, axis=0,)
                            >= n_oar_pred
                        ):
                            out.append("Better")
                        else:
                            out.append("No Change")
                out.insert(0, str(case))
                results.append(out)
            predicted_results = pd.DataFrame(results)
            predicted_results.iloc[7, 4] = None  # Handle case 78
            predicted_results.iloc[10, 4] = None  # Handle case 82
            predicted_results.to_csv(
                os.path.join(
                    output_path,
                    "predicted_results_" + str(alpha) + "_" + str(n_oar_pred) + ".csv",
                )
            )

            predicted_results = predicted_results.loc[:, 1:].stack().reset_index()

            data_dict = {
                "Pred": predicted_results,
                "R1": jw_results,
                "R2": ee_results,
                "R3": er_results,
            }
            cat_names = ["Better", "No Change", "Worse"]
            for key in data_dict.keys():
                coeff = spearmanr(
                    list(reference_results.iloc[:, 2]), list(data_dict[key].iloc[:, 2]),
                )

                score = ndcg_score(
                    np.asarray(
                        [
                            list(
                                reference_results.iloc[:, 2]
                                .astype("category")
                                .cat.codes
                            )
                        ]
                    ),
                    np.asarray(
                        [list(data_dict[key].iloc[:, 2].astype("category").cat.codes)]
                    ),
                )
                print(
                    classification_report(
                        list(reference_results.iloc[:, 2]),
                        list(data_dict[key].iloc[:, 2]),
                        labels=cat_names,
                    )
                )
                print(f"Spearman coefficient (GT vs {key}) is: {coeff}\n")
                print(f"nDCG for (GT vs {key}) is: {score}\n")
                if alpha not in precision_map[key].keys():
                    precision_map[key][alpha] = {}
                precision_map[key][alpha][n_oar_pred] = precision_score(
                    list(reference_results.iloc[:, 2]),
                    list(data_dict[key].iloc[:, 2]),
                    average="weighted",
                )
                if alpha not in recall_map[key].keys():
                    recall_map[key][alpha] = {}
                recall_map[key][alpha][n_oar_pred] = recall_score(
                    list(reference_results.iloc[:, 2]),
                    list(data_dict[key].iloc[:, 2]),
                    average="weighted",
                )
                if alpha not in f1_map[key].keys():
                    f1_map[key][alpha] = {}
                f1_map[key][alpha][n_oar_pred] = f1_score(
                    list(reference_results.iloc[:, 2]),
                    list(data_dict[key].iloc[:, 2]),
                    average="weighted",
                )
    precision_df = pd.DataFrame.from_dict(precision_map["Pred"])
    precision_array = precision_df.to_numpy()
    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(
        precision_array,
        #mask=precision_array < 0.48,
        xticklabels=precision_df.columns,
        yticklabels=precision_df.index,
        cmap="viridis",
        annot=True,
        square=True,
        cbar=False,
    )
    plt.title(r"Variation of Precision (to $\alpha$ and nOAR)")
    plt.gca().set_xlabel(r"$\alpha$")
    plt.ylabel("nOAR")
    plt.show()

    recall_df = pd.DataFrame.from_dict(recall_map["Pred"])
    recall_array = recall_df.to_numpy()
    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(
        recall_array,
        #mask=recall_array < 0.46,
        xticklabels=recall_df.columns,
        yticklabels=recall_df.index,
        cmap="viridis",
        annot=True,
        square=True,
        vmin=0.3,
        vmax=0.75,
        cbar=False,
    )
    plt.title(r"Variation of Recall (to $\alpha$ and nOAR)")
    plt.gca().set_xlabel(r"$\alpha$")
    plt.ylabel("nOAR")
    plt.show()

    f1_df = pd.DataFrame.from_dict(f1_map["Pred"])
    f1_array = f1_df.to_numpy()
    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(
        f1_array,
        #mask=f1_array < 0.44,
        xticklabels=f1_df.columns,
        yticklabels=f1_df.index,
        cmap="viridis",
        annot=True,
        square=True,
        vmin=0.3,
        vmax=0.75,
        cbar=False,
    )
    plt.title(r"Variation of F1 score (to $\alpha$ and nOAR)")
    plt.gca().set_xlabel(r"$\alpha$")
    plt.ylabel("nOAR")
    plt.show()
