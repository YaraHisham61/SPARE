def xyxy_to_xywh_converter(bbox):
    """
    bbox: [x1, y1, x2, y2]
    returns: [x, y, w, h]
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]

def convert_boxes_to_xywh(bboxes_xyxy):
    return [xyxy_to_xywh_converter(b) for b in bboxes_xyxy]