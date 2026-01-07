from lib.parsing.model.parsing_result import ParsingBoundingBox


def merge_adjacent_boxes(
    geom: list[ParsingBoundingBox],
    max_y_distance: float = 0.1,
    min_x_overlap: float = 0.5
) -> list[ParsingBoundingBox]:
    """
    Column aware merging of adjacent bounding boxes.

    Args:
        geom: List of Bounding Boxes to be merged
        max_y_distance: Maximum distance between two elements so they are considered adjacent.
        Fractional of page height. Default: 0.05
        min_x_overlap: Threshold of width overlap of two elements so they are considered adjacent.
        Fractional of the smaller element's width. Default: 0.5

    Returns:
        List of ParsingBoundingBox where adjacent boxes are merged into their unions
    """

    merged_boxes: list[ParsingBoundingBox] = []

    for box in geom:
        i = 0
        while i < len(merged_boxes):
            merged_box = merged_boxes[i]

            smallest_width = min(box.right - box.left, merged_box.right - merged_box.left)
            intersect_start = max(box.left, merged_box.left)
            intersect_end = min(box.right, merged_box.right)

            x_overlap = (intersect_end - intersect_start) / smallest_width
            y_distance = merged_box.top - box.bottom

            if x_overlap >= min_x_overlap and y_distance <= max_y_distance:
                # box becomes the union of both boxes
                box.left = min(box.left, merged_box.left)
                box.top = min(box.top, merged_box.top)
                box.right = max(box.right, merged_box.right)
                box.bottom = max(box.bottom, merged_box.bottom)

                # We no longer need the absorbed box
                merged_boxes.remove(merged_box)
                # After the merge we might now be able to merge with previously unmatching boxes
                i = 0

            else:
                i += 1

        merged_boxes.append(box)

    return merged_boxes
