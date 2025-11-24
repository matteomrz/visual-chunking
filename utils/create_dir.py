from pathlib import Path


def create_directory(
    file_path: Path,
    src_dir: Path,
    dst_dir: Path,
    with_file: bool = False
) -> Path:
    """
        Helper method to create a helper directory for an input file
        at the given dst directory.

        Example:
            ``<src_dir>/<some_path>/<file>.*``
            -> ``<dst_dir>/<some_path>/[optional: <file>]``

        Args:
            file_path: The absolute path to the input PDF file
            src_dir: The directory which contains the original file.
            dst_dir: The directory which contains the created directories.
            with_file: If True, will create a directory with the name of the file.

        Returns:
            Path of the created directory
    """
    parents = file_path.relative_to(src_dir).parents
    final_dir = dst_dir

    # Handle batch names
    for parent in parents:
        final_dir = final_dir / parent
    if with_file:
        final_dir = final_dir / file_path.stem

    if not final_dir.exists():
        final_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory at: {final_dir}")

    return final_dir
