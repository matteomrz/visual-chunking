from pathlib import Path


def create_directory(
    file_path: Path, src_dir: Path, dst_dir: Path, with_file: bool = False
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
    final_dir = get_directory(file_path, src_dir, dst_dir, with_file)

    if not final_dir.exists():
        final_dir.mkdir(parents=True, exist_ok=True)
        print(f"Info: Created directory at: {final_dir}")
    else:
        print(f"Info: Directory already exists: {final_dir}")

    return final_dir


def get_directory(
    file_path: Path, src_dir: Path, dst_dir: Path, with_file: bool = False
) -> Path:
    """
    Helper method to get the path of a helper directory for an input file
    at the given dst directory specifically without creating it.
    Usually create_directory should be preferred.

    Example:
        ``<src_dir>/<some_path>/<file>.*``
        -> ``<dst_dir>/<some_path>/[optional: <file>]``

    Args:
        file_path: The absolute path to the input PDF file
        src_dir: The directory which contains the original file.
        dst_dir: The directory which contains the created directories.
        with_file: If True, will create a directory with the name of the file.

    Returns:
        Path of the directory without guarantee for existence
    """
    parent = file_path.relative_to(src_dir).parent
    final_dir = dst_dir / parent

    if with_file:
        final_dir = final_dir / file_path.stem

    return final_dir
