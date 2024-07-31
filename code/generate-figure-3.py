import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    ndcg_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

import matplotlib

font = {"size": 12}
matplotlib.rc("font", **font)


def rank_scores(y_true, y_score):
    score = ndcg_score(np.asarray([y_true]), np.asarray([y_score]))
    coeff = spearmanr(y_true, y_score)
    print(f"Between {y_true} and {y_score}:")
    print(f"nDCG score is {score}")
    print(f"Spearman rank coefficient is {coeff.statistic}")
    return score, coeff


def generate_reference_results(cases, list_oar_names, type, thresh_gt, n_oar_gt, root_path):

    output_path = os.path.join(
        os.path.join(root_path, "results"),
        str(thresh_gt) + "_" + str(n_oar_gt) + "_" + type,
    )
    os.makedirs(output_path, exist_ok=True)

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
    return reference_results


def generate_predicted_results(cases, list_oar_names, type, alpha, thresh_pred, n_oar_pred, root_path):

    output_path = os.path.join(
        os.path.join(root_path, "results"),
        str(thresh_pred) + "_" + str(n_oar_pred) + "_" + type,
    )
    os.makedirs(output_path, exist_ok=True)
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
                data[["1", "2", "3", "4"]].div(data["0"], axis=0).loc[list_oar_names]
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
    return predicted_results


if __name__ == "__main__":

    file_path = os.path.dirname(__file__)
    path_to_data = os.path.join(file_path, '../radonc-vs-dldp-data.zip')
    zf = zipfile.ZipFile(path_to_data)

    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)

        expert_path = os.path.join(tempdir, "radonc-vs-dldp-data", "expert")
        data_path = os.path.join(tempdir, "radonc-vs-dldp-data", "experiments")
        reference_path = os.path.join(data_path, "reference")
        prediction_path = os.path.join(data_path, "predicted")

        jw_results = pd.read_csv(os.path.join(expert_path, "jw_impact.csv"))
        jw_results = jw_results.iloc[:, [1, 2, 3, 4]].stack().reset_index()

        ee_results = pd.read_csv(os.path.join(expert_path, "ee_impact.csv"))
        ee_results = ee_results.iloc[:, [1, 2, 3, 4]].stack().reset_index()

        er_results = pd.read_csv(os.path.join(expert_path, "er_impact.csv"))
        er_results = er_results.iloc[:, [1, 2, 3, 4]].stack().reset_index()

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

        cases = [70, 71, 72, 73, 74, 75, 76, 78, 80, 81, 82, 83, 84, 85, 86]
        type = "max"
        n_oar_gt = 1
        thresh_gt = 0.1
        reference_results = generate_reference_results(cases, list_oar_names, type, thresh_gt, n_oar_gt, tempdir)

        alpha = 0.1
        n_oar_pred = 3
        thresh_pred = alpha * thresh_gt
        predicted_results = generate_predicted_results(cases, list_oar_names, type, alpha, thresh_pred, n_oar_pred, tempdir)

        data_dict = {
            "Prediction": predicted_results,
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
                    [list(reference_results.iloc[:, 2].astype("category").cat.codes)]
                ),
                np.asarray([list(data_dict[key].iloc[:, 2].astype("category").cat.codes)]),
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
            conf_mat = confusion_matrix(
                list(reference_results.iloc[:, 2]),
                list(data_dict[key].iloc[:, 2]),
                normalize="true",
            )
            print(f"Confusion matrix for (Reference vs {key}) is: \n{conf_mat}")
            # plt.figure(figsize=(24, 24))
            # cm = ConfusionMatrixDisplay(conf_mat, display_labels=cat_names)
            # cm.plot()
            # plt.title(f"Reference vs {key}")
            # plt.show()
            df_cm = pd.DataFrame(conf_mat, index=cat_names, columns=cat_names)
            plt.figure(figsize=(24, 24))
            # cm = ConfusionMatrixDisplay(conf_mat, display_labels=cat_names)
            # cm.plot()
            off_diag_mask = np.eye(*conf_mat.shape, dtype=bool)

            vmin = np.min(conf_mat)
            vmax = np.max(conf_mat)
            fig = plt.figure()
            sns.heatmap(
                df_cm, annot=True, mask=~off_diag_mask, cmap="Greens", vmin=vmin, vmax=vmax
            )
            sns.heatmap(
                df_cm,
                annot=True,
                mask=off_diag_mask,
                cmap="OrRd",
                vmin=vmin,
                vmax=vmax,
                cbar_kws=dict(ticks=[]),
            )
            plt.title(f"Reference vs {key}")
            plt.show()
