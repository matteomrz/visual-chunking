from logging import getLogger

from pandas import DataFrame

from config import TABLE_DIR

logger = getLogger(__name__)


def export_table_to_latex(
    df: DataFrame,
    name: str,
    axis: int = 0,
    switch_highlighting: bool = False,
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
        switch_highlighting: whether to switch the highlighting style of max and min values. Default: False
        precision: floating point precision used in the output. Default: 4
    """

    styler = df.style

    bold_style = "bfseries:;"  # \bfseries
    underline_style = "underline:--rwrap;"  # \underline{}

    if not switch_highlighting:
        styler.highlight_max(axis=axis, props=bold_style)
        styler.highlight_min(axis=axis, props=underline_style)
    else:
        styler.highlight_min(axis=axis, props=bold_style)
        styler.highlight_max(axis=axis, props=underline_style)

    # Escape characters in headers
    styler.format_index(escape="latex", axis=1)
    styler.format_index(escape="latex", axis=0)
    styler.format(precision=precision)

    tex_content = styler.to_latex(hrules=True)

    if not TABLE_DIR.exists():
        TABLE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLE_DIR / f"{name}.tex"

    with open(output_path, "w") as f:
        f.write(tex_content)
        logger.info(f"Saved table content to: {output_path}")
