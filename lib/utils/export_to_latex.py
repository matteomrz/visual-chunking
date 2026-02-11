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
    replace_zeros: str | None = None,
    switch_highlighting: bool | list[bool] = False,
    escape_latex=True,
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
        replace_zeros: character to replace zero entries with. Default: No replacement
        switch_highlighting: whether to switch the highlighting style of max and min values. \
         Can either be a single value for all columns or a list of booleans containing a value for each individual columns. \
         If the length of the list does not match the number of columns, it will use the default for all columns. \
         Default: False
        escape_latex: whether latex text should be escaped in index and column headers. Default: True
        precision: floating point precision used in the output. Default: 4
    """

    columns = df.columns.tolist()

    highlight_max = []
    highlight_min = []

    if isinstance(switch_highlighting, list):
        if not len(switch_highlighting) == len(columns):
            logger.warning(
                "Columnwise highlighting information is incomplete. "
                f"Expected {len(columns)}. Actual: {len(switch_highlighting)}. "
                "Highlighting max values in all columns."
            )
            highlight_max = columns

        else:
            for i in range(len(switch_highlighting)):
                if switch_highlighting[i]:
                    highlight_min.append(columns[i])
                else:
                    highlight_max.append(columns[i])

    elif switch_highlighting:
        highlight_min = columns
    else:
        highlight_max = columns

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
        rep_zeros = lambda num: replace_zeros if num == 0 else round(num, 4)
        styler.format(rep_zeros)
    else:
        styler.format(precision=precision)

    tex_content = styler.to_latex(hrules=True)

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
