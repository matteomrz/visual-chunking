import pandas as pd

from lib.chunking.methods.chunkers import Chunkers
from lib.utils.export_table import export_table_to_latex
from lib.utils.thesis_names import get_chunker_name


def _combine_cols(df, metric_name: str):
    mean_col = f"{metric_name}_mean"
    std_col = f"{metric_name}_std"

    sorted_mean = df[mean_col].round(2).unique()
    sorted_mean.sort()

    max_val = sorted_mean[-1]
    second_val = sorted_mean[-2]

    def highlight_str(row):
        mean = round(row[mean_col], 2)
        std = round(row[std_col], 2)
        cell = fr"{mean:.2f} \pm {std:.2f}"

        if mean == max_val:
            return fr"$\mathbf{{{cell}}}$"
        elif mean == second_val:
            return fr"\underline{{${cell}$}}"
        else:
            return f"${cell}$"

    df[metric_name.capitalize()] = df.apply(highlight_str, axis=1)


def export_results(df: pd.DataFrame, export_name: str) -> pd.DataFrame:
    df = df.copy()
    method_order = [get_chunker_name(m) for m in Chunkers]
    df["Method"] = pd.Categorical(df["Method"], categories=method_order, ordered=True)

    df.sort_values(by=["Method", "N", "Param"], inplace=True)
    df.set_index(["Method", "Param"], inplace=True)
    df.index.names = [None, None]

    _combine_cols(df, "iou")
    _combine_cols(df, "precision")
    _combine_cols(df, "recall")
    _combine_cols(df, "precision_omega")
    df = df[["N", "Iou", "Recall", "Precision", "Precision_omega"]]

    df = df.rename(columns={
        "Iou": "IoU",
        "Precision_omega": r"$\text{Precision}_\Omega$"
    })

    export_table_to_latex(
        df,
        name=export_name,
        sort_by_index=False,
        column_format="llccccc",
        escape_latex=False,
        add_lines_between_index=True,
        highlight_mode=None  # Highlighting was handled by _combine_cols
    )

    return df
