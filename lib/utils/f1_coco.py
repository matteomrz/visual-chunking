import logging

from faster_coco_eval import COCOeval_faster
import numpy as np

logger = logging.getLogger(__name__)


def get_f1_metrics(coco_eval: COCOeval_faster) -> list[dict]:
    """
    Logic adapted from COCOEval_faster.extended_metrics().
    Calculates per-class and overall F1 metrics at IoU thresholds 0.5 and 0.5:95.

    Args:
        coco_eval: COCOeval_faster object after running evaluate()

    Returns:
        list[dict]: Results for every category with the following keys:
            - 'class' (str): Class name or "all" for macro metrics.
            - 'f1@50:95' (float): F1 score at IoU 0.50:0.95.
            - 'f1@50' (float): F1 score at IoU 0.50.
    """

    recalls = coco_eval.params.recThrs

    # shape: [I, R, C]
    # One precision value for each (C)lass at (R)ecall and (I)oU
    # Area size is set to all (i:0) and maxDets is set to 100 (i:-1)
    raw_precisions = coco_eval.eval["precision"][:, :, :, 0, -1]

    precisions = raw_precisions.copy().astype(float)
    precisions[precisions < 0] = np.nan

    # Match Recall shape to [I, R, C]
    recall_grid = recalls[None, :, None]

    f1_numerator = 2 * precisions * recall_grid
    f1_denominator = precisions + recall_grid

    f1_raw = np.divide(
        f1_numerator,
        f1_denominator,
        out=np.zeros_like(f1_numerator),
        where=f1_denominator != 0
    )

    best_f1_per_iou_class = np.nanmax(f1_raw, axis=1)

    iou_thrs = coco_eval.params.iouThrs
    iou_50_idx = np.argwhere(np.isclose(iou_thrs, 0.50)).item()

    per_class = []

    cat_ids = coco_eval.params.cat_ids
    cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}

    for k, cid in enumerate(cat_ids):
        cat_name = cat_id_to_name[cid]

        f1_per_iou = best_f1_per_iou_class[:, k]

        f1_50 = f1_per_iou[iou_50_idx].item()
        f1_50_95 = np.nanmean(f1_per_iou).item()

        if np.isnan(f1_50) or np.isnan(f1_50_95):
            logger.warning(
                f"Invalid F1 scores for {cat_name}. "
                f"F1@50={f1_50}, F1@50:95={f1_50_95}"
            )
            continue

        per_class.append({
            "class": cat_name,
            "f1@50:95": f1_50_95,
            "f1@50": f1_50,
        })

    # f1 per iou and recall value averaged through classes
    f1_macro = np.nanmean(f1_raw, axis=2)
    best_f1_per_iou_overall = np.nanmax(f1_macro, axis=1)

    f1_all_50 = best_f1_per_iou_overall[iou_50_idx].item()
    f1_all_50_95 = np.nanmean(best_f1_per_iou_overall).item()

    per_class.append({
        "class": "all",
        "f1@50:95": f1_all_50_95,
        "f1@50": f1_all_50,
    })

    return per_class
