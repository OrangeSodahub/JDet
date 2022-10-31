import jittor as jt


def multiclass_nms_rotated(multi_bboxes,
                           multi_scores,
                           score_thr,
                           nms,
                           max_num=-1,
                           score_factors=None,
                           return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (jittor.Var): shape (n, #class*5) or (n, 5)
        multi_scores (jittor.Var): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): Config of NMS.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple (dets, labels, indices (optional)): tensors of shape (k, 5), \
        (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 5)
    scores = multi_scores[:, :-1]

    labels = jt.arange(num_classes, dtype=jt.long)
    labels = labels.view(1, -1).expand_as(scores)
    bboxes = bboxes.reshape(-1, 5)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    # remove low scoring boxes
    valid_mask = scores > score_thr
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    if bboxes.numel() == 0:
        dets = jt.concat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    # Strictly, the maximum coordinates of the rotating box (x,y,w,h,a)
    # should be calculated by polygon coordinates.
    # But the conversion from rbbox to polygon will slow down the speed.
    # So we use max(x,y) + max(w,h) as max coordinate
    # which is larger than polygon max coordinate
    # max(x1, y1, x2, y2,x3, y3, x4, y4)
    max_coordinate = bboxes[:, :2].max() + bboxes[:, 2:4].max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    if bboxes.size(-1) == 5:
        bboxes_for_nms = bboxes.clone()
        bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
    else:
        bboxes_for_nms = bboxes + offsets[:, None]
    _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if return_inds:
        return jt.concat([bboxes, scores[:, None]], 1), labels, keep
    else:
        return jt.concat([bboxes, scores[:, None]], 1), labels


def nms_rotated(dets: Tensor,
                scores: Tensor,
                iou_threshold: float,
                labels: Optional[Tensor] = None,
                clockwise: bool = True) -> Tuple[Tensor, Tensor]:
    """Performs non-maximum suppression (NMS) on the rotated boxes according to
    their intersection-over-union (IoU).

    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.

    Args:
        dets (torch.Tensor):  Rotated boxes in shape (N, 5).
            They are expected to be in
            (x_ctr, y_ctr, width, height, angle_radian) format.
        scores (torch.Tensor): scores in shape (N, ).
        iou_threshold (float): IoU thresh for NMS.
        labels (torch.Tensor, optional): boxes' label in shape (N,).
        clockwise (bool): flag indicating whether the positive angular
            orientation is clockwise. default True.
            `New in version 1.4.3.`

    Returns:
        tuple: kept dets(boxes and scores) and indice, which is always the
        same data type as the input.
    """
    if dets.shape[0] == 0:
        return dets, None
    if not clockwise:
        flip_mat = dets.new_ones(dets.shape[-1])
        flip_mat[-1] = -1
        dets_cw = dets * flip_mat
    else:
        dets_cw = dets
    multi_label = labels is not None
    if multi_label:
        dets_wl = torch.cat((dets_cw, labels.unsqueeze(1)), 1)  # type: ignore
    else:
        dets_wl = dets_cw
    _, order = scores.sort(0, descending=True)
    dets_sorted = dets_wl.index_select(0, order)

    if torch.__version__ == 'parrots':
        keep_inds = ext_module.nms_rotated(
            dets_wl,
            scores,
            order,
            dets_sorted,
            iou_threshold=iou_threshold,
            multi_label=multi_label)
    else:
        keep_inds = ext_module.nms_rotated(dets_wl, scores, order, dets_sorted,
                                           iou_threshold, multi_label)
    dets = torch.cat((dets[keep_inds], scores[keep_inds].reshape(-1, 1)),
                     dim=1)
    return dets, keep_inds