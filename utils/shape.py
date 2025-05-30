# MIT License
# Copyright (c) Kentaro Wada

import math
import uuid

import numpy as np
import PIL.Image
import PIL.ImageDraw

import logging
from tqdm import tqdm


def polygons_to_mask(img_shape, polygons, shape_type=None):
	logging.warning(
		"The 'polygons_to_mask' function is deprecated, " "use 'shape_to_mask' instead."
	)
	return shape_to_mask(img_shape, points=polygons, shape_type=shape_type)


def shape_to_mask(img_shape, points, shape_type=None, line_width=10, point_size=5):
	mask = np.zeros(img_shape[:2], dtype=np.uint8)
	mask = PIL.Image.fromarray(mask)
	draw = PIL.ImageDraw.Draw(mask)
	xy = [tuple(point) for point in points]
	if shape_type == "circle":
		assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
		(cx, cy), (px, py) = xy
		d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
		draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
	elif shape_type == "rectangle":
		assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
		draw.rectangle(xy, outline=1, fill=1)
	elif shape_type == "line":
		assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
		draw.line(xy=xy, fill=1, width=line_width)
	elif shape_type == "linestrip":
		draw.line(xy=xy, fill=1, width=line_width)
	elif shape_type == "point":
		assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
		cx, cy = xy[0]
		r = point_size
		draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
	else:
		assert len(xy) > 2, "Polygon must have points more than 2"
		draw.polygon(xy=xy, outline=1, fill=1)
	mask = np.array(mask, dtype=bool)
	return mask


def shapes_to_label(img_shape, shapes, label_name_to_value):
	# cls = np.zeros(img_shape[:2], dtype=np.int32)
	cls = np.full(img_shape[:2], 2, dtype=np.int8)
	for shape in tqdm(shapes, desc="Converting shapes to masks"):
		points = shape["points"]
		label = shape["label"]
		shape_type = shape.get("shape_type", None)

		cls_name = label

		cls_id = label_name_to_value[cls_name]

		mask = shape_to_mask(img_shape[:2], points, shape_type)
		cls[mask] = cls_id

	return cls


def labelme_shapes_to_label(img_shape, shapes):
	logging.warning(
		"labelme_shapes_to_label is deprecated, so please use " "shapes_to_label."
	)

	label_name_to_value = {"_background_": 0}
	for shape in shapes:
		label_name = shape["label"]
		if label_name in label_name_to_value:
			label_value = label_name_to_value[label_name]
		else:
			label_value = len(label_name_to_value)
			label_name_to_value[label_name] = label_value

	lbl = shapes_to_label(img_shape, shapes, label_name_to_value)
	return lbl, label_name_to_value


def masks_to_bboxes(masks):
	if masks.ndim != 3:
		raise ValueError("masks.ndim must be 3, but it is {}".format(masks.ndim))
	if masks.dtype != bool:
		raise ValueError(
			"masks.dtype must be bool type, but it is {}".format(masks.dtype)
		)
	bboxes = []
	for mask in masks:
		where = np.argwhere(mask)
		(y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
		bboxes.append((y1, x1, y2, x2))
	bboxes = np.asarray(bboxes, dtype=np.float32)
	return bboxes
