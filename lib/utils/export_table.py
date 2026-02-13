from functools import partial

import numpy as np
from logging import getLogger

from pandas import DataFrame, Series

from config import PLOT_DIR, TABLE_DIR

logger = getLogger(__name__)


def _get_second(df: Series, is_max: bool, props: str):
    """Get secondary highlight elements."""
    unique = df.dropna().unique()
    unique.sort()

    if len(unique) < 2:
        return np.full(df.shape, "", dtype=object)

    if is_max:
        val = unique[-2]
    else:
        val = unique[1]

    return np.where(df == val, props, "")


def export_table_to_latex(
    df: DataFrame,
    name: str,
    axis: int = 0,
    column_format: str | None = None,
    replace_zeros: str | None = None,
    highlight_mode: bool | list[bool | None] | None = True,
    escape_latex=True,
    sort_by_index=True,
    precision: int = 4
):
    """
    Saves the content from a DataFrame to 'TABLE_DIR / ```name```.tex'.
    Automatically highlights the max and min values.
    Max values are bolded and min values are underlined by default.

    Args:
        df: pandas.DataFrame to convert
        name: name of the resulting .tex file
        axis: on which axis the max value should be highlighted. Default: 0 (0: column, 1: row)
        column_format: \begin{tabular}{<column_format>}. Default: None
        replace_zeros: character to replace zero entries with. Default: No replacement
        highlight_mode: three different highlighting modes available. \
         True: highlight max as bold, second-highest underscore. \
         False: highlight min as bold, second-lowest underscore. \
         None: no highlighting. \
         Can also be a list of values for each column.  \
         If the length of the list does not match the number of columns, it will use the default for all columns. \
         Default: True (Highlight max)
        escape_latex: whether latex text should be escaped in index and column headers. Default: True
        sort_by_index: whether to sort the dataframe by the index before exporting. Default: True
        precision: floating point precision used in the output. Default: 4
    """

    if sort_by_index:
        df.sort_index(inplace=True)

    columns = df.columns.tolist()

    highlight_max = []
    highlight_min = []

    if isinstance(highlight_mode, list):
        if not len(highlight_mode) == len(columns):
            logger.warning(
                "Columnwise highlighting information is incomplete. "
                f"Expected {len(columns)}. Actual: {len(highlight_mode)}. "
                "Highlighting max values in all columns."
            )
            highlight_max = columns

        else:
            for i in range(len(highlight_mode)):
                if highlight_mode[i]:
                    highlight_max.append(columns[i])
                elif highlight_mode[i] is not None:
                    highlight_min.append(columns[i])

    elif highlight_mode:
        highlight_max = columns
    elif highlight_mode is not None:
        highlight_min = columns

    styler = df.style

    bold_style = "bfseries:;"  # \bfseries
    underline_style = "underline:--rwrap;"  # \underline{}

    if highlight_max:
        styler.highlight_max(axis=axis, props=bold_style, subset=highlight_max)
        styler.apply(
            partial(_get_second, is_max=True),
            axis=axis,
            props=underline_style,
            subset=highlight_max
        )

    if highlight_min:
        styler.highlight_min(axis=axis, props=bold_style, subset=highlight_min)
        styler.apply(
            partial(_get_second, is_max=False),
            axis=axis,
            props=underline_style,
            subset=highlight_min
        )

    # Escape characters in headers
    if escape_latex:
        styler.format_index(escape="latex", axis=1)
        styler.format_index(escape="latex", axis=0)

    # Replace zeros
    if replace_zeros:
        rep_zeros = lambda num: replace_zeros if num == 0 else f"{num:.4f}"
        styler.format(rep_zeros)
    else:
        styler.format(precision=precision)

    tex_content = styler.to_latex(
        hrules=True,
        column_format=column_format,
        multicol_align="c"
    )

    if not TABLE_DIR.exists():
        TABLE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLE_DIR / f"{name}.tex"

    with open(output_path, "w") as f:
        f.write(tex_content)
        logger.info(f"Saved table content to: {output_path}")


def export_table_to_csv(df: DataFrame, name: str):
    """The exported data is saved as a csv file to 'PLOT_DIR / ```name```.csv'."""

    if not PLOT_DIR.exists():
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = PLOT_DIR / f"{name}.csv"
    df.to_csv(path_or_buf=csv_path, index=False)
    logger.info(f"Saved table data to: {csv_path}")
