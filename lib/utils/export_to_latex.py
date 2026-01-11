from logging import getLogger

from pandas import DataFrame

from config import FIGURE_DIR

logger = getLogger(__name__)


def export_table_to_latex(df: DataFrame, name: str, axis: int = 0, precision: int = 4):
    """
    Saves the content from a DataFrame to 'FIGURE_DIR / ```name```.tex'.
    Automatically highlights the max values.

    Args:
        df: pandas.DataFrame to convert
        name: name of the resulting .tex file
        axis: on which axis the max value should be highlighted. Default: 0 (0: column, 1: row)
        precision: floating point precision used in the output. Default: 4
    """

    styler = df.style

    styler.highlight_max(axis=axis, props="font-weight:bold;")
    # Escape characters in headers
    styler.format_index(escape="latex", axis=1)
    styler.format_index(escape="latex", axis=0)
    styler.format(precision=4)

    tex_content = styler.to_latex(hrules=True, convert_css=True)

    if not FIGURE_DIR.exists():
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURE_DIR / f"{name}.tex"

    with open(output_path, "w") as f:
        f.write(tex_content)
        logger.info(f"Saved table content to: {output_path}")
