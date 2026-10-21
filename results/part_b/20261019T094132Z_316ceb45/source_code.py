"""Stereo Venus: hình học epipolar, disparity BM/SGBM và point cloud."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import shutil
import struct
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

DEFAULT_INPUT = Path("data/part_b/Venus")
DEFAULT_OUTPUT = Path("results/part_b")
DEFAULT_CONFIG = {
    "profile": "full", "methods": ["bm", "sgbm"],
    "num_disparities": [32, 64], "block_sizes": [5, 9, 15],
    "min_disparity": -16, "ransac_thresholds_px": [0.5, 1.0, 2.0],
    "max_features": 5000, "ratio": 0.75, "seed": 42,
    "confidence": 0.999, "max_iters": 5000,
    "lr_tolerance_px": 1.0, "bad_pixel_threshold_px": 1.0,
    "warmup": 3, "repetitions": 20,
}
SMOKE_CONFIG = dict(DEFAULT_CONFIG, profile="smoke", num_disparities=[32],
                    block_sizes=[9], ransac_thresholds_px=[1.0], warmup=0, repetitions=1)
GEOMETRY_PROTOCOL = {"estimator": "cv2.USAC_MAGSAC (RANSAC family)", "native_random_state": 0,
    "local_optimization": "native sigma-consensus preset", "uses_ground_truth_for_estimation": False,
    "diagnostic_angle_p95_limit_deg": 2.0}
INPUT_HASHES = {
    "frame10.png": "f36d77900848c4d5bf6f31898e43d3db4c912f474f273b331d7a96113ea5a637",
    "frame11.png": "45860656c48fec7fff1ed3528bd2d72f4f6dbf71c6d621fb2e0f48a531e617d0",
    "flow10.flo": "4f5e58609d02d8198f838de8b3f34a952cfaebf284938daa255066c535610f34",
}
REFERENCES = {
    "Baker2007": {
        "authors": ["Simon Baker", "Daniel Scharstein", "J. P. Lewis", "Stefan Roth", "Michael J. Black", "Richard Szeliski"],
        "title": "A Database and Evaluation Methodology for Optical Flow",
        "year": 2007, "conference": "IEEE International Conference on Computer Vision",
        "doi": "10.1109/ICCV.2007.4408903",
        "url": "https://cs.brown.edu/people/mjblack/Papers/flowEval07.pdf",
        "section": "3.4 and Figure 6", "used_for": "Venus modified stereo data and crop-induced signed flow.",
    },
    "OpenCVUSAC": {
        "url": "https://docs.opencv.org/4.x/de/d3e/tutorial_usac.html",
        "source": "https://github.com/opencv/opencv/blob/5.0.0/modules/geometry/src/usac/ransac_solvers.cpp",
        "used_for": "USAC_MAGSAC RANSAC preset with native sigma-consensus local optimization; native random state 0.",
    },
    "Basha2013": {
        "authors": ["Tali Basha", "Yael Moses", "Nahum Kiryati"],
        "title": "Multi-view Scene Flow Estimation: A View Centered Variational Approach",
        "year": 2013, "journal": "International Journal of Computer Vision",
        "volume": 101, "pages": "6–21", "doi": "10.1007/s11263-012-0542-7",
        "url": "https://link.springer.com/article/10.1007/s11263-012-0542-7",
        "author_pdf": "https://people.csail.mit.edu/talidekel/papers/MVSF_IJCV13.pdf",
        "section": "4.1 (page 9 of the author PDF)",
        "used_for": "30 degree viewing angle and focal-length construction for the pinhole camera model.",
    },
    "Baker2011": {
        "authors": ["Simon Baker", "Daniel Scharstein", "J. P. Lewis", "Stefan Roth", "Michael J. Black", "Richard Szeliski"],
        "title": "A Database and Evaluation Methodology for Optical Flow",
        "year": 2011, "journal": "International Journal of Computer Vision",
        "volume": 92, "pages": "1–31", "doi": "10.1007/s11263-010-0390-2",
        "url": "https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf",
        "used_for": "Flow dataset provenance; modified stereo images and shifted disparity ranges.",
    },
    "Scharstein2002": {
        "authors": ["Daniel Scharstein", "Richard Szeliski"],
        "title": "A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms",
        "year": 2002, "journal": "International Journal of Computer Vision",
        "volume": 47, "pages": "7–42",
        "url": "https://vision.middlebury.edu/stereo/taxonomy-IJCV.pdf",
        "section": "5.2", "used_for": "Original Venus acquisition, rectification and disparity coordinate convention.",
    },
    "MiddleburyFlow": {
        "url": "https://vision.middlebury.edu/flow/data/",
        "format_url": "https://vision.middlebury.edu/flow/code/flow-code/README.txt",
        "used_for": "Official input archives, .flo layout and unknown-flow threshold.",
    },
    "OpenCV5": {
        "url": "https://github.com/opencv/opencv/tree/5.0.0/modules/stereo",
        "used_for": "Native BM/SGBM, fixed-point disparities and effective disp12MaxDiff behavior.",
    },
    "OpenCVGeometry": {
        "url": "https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html",
        "used_for": "findFundamentalMat, epipolar lines and reprojectImageTo3D API conventions; APIs checked on installed OpenCV 5.0.0.",
    },
}


def _digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root phải là object: {path}")
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_flow(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    """Kiểm tra payload trước khi chuyển file .flo cho bộ đọc native."""
    with Path(path).open("rb") as stream:
        header = stream.read(12)
    if len(header) != 12:
        raise ValueError("Header .flo bị thiếu.")
    tag, width, height = struct.unpack("<fii", header)
    if tag != 202021.25 or width <= 0 or height <= 0 or (width, height) != expected_size:
        raise ValueError("Header .flo không khớp định dạng/kích thước ảnh.")
    if Path(path).stat().st_size != 12 + width * height * 8:
        raise ValueError("Độ dài payload .flo không hợp lệ.")
    flow = cv2.readOpticalFlow(str(path))
    if flow is None or flow.shape != (height, width, 2):
        raise ValueError("OpenCV không giải mã được flow.")
    return flow


def flow_ground_truth(flow: np.ndarray) -> dict:
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("Flow phải có shape H×W×2.")
    valid = np.isfinite(flow).all(axis=2) & (np.abs(flow) <= 1e9).all(axis=2)
    if not valid.any():
        raise ValueError("Flow không có pixel GT hợp lệ.")
    if np.max(np.abs(flow[:, :, 1][valid])) > 1e-4:
        raise ValueError("Flow dọc khác 0: không dùng GT này cho stereo cùng hàng.")
    height, width = valid.shape
    y, x = np.indices(valid.shape)
    target_x = x + flow[:, :, 0]
    target_y = y + flow[:, :, 1]
    in_bounds = (target_x >= 0) & (target_x <= width - 1) & (target_y >= 0) & (target_y <= height - 1)
    return {"disparity": -flow[:, :, 0].astype(np.float32), "valid": valid,
            "evaluation_domain": valid & in_bounds}


def load_stereo_pair(input_dir: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Camera giả định chỉ áp dụng cho đúng các file Venus đã xác minh crop."""
    input_dir = Path(input_dir)
    metadata = _read_json(input_dir / "metadata.json")
    if not isinstance(metadata.get("files"), dict) or not isinstance(metadata.get("stereo_geometry"), dict):
        raise ValueError("Metadata phải chứa các object files và stereo_geometry.")
    if metadata.get("scene") != "Venus":
        raise ValueError("Pipeline này dùng riêng cặp Venus đã xác minh.")
    for name, expected_hash in INPUT_HASHES.items():
        record = metadata.get("files", {}).get(name, {})
        if not isinstance(record, dict):
            raise ValueError(f"Metadata file không phải object: {name}")
        if record.get("sha256") != expected_hash or _digest(input_dir / name) != expected_hash:
            raise ValueError(f"SHA-256 không khớp Venus: {name}")
        if (record.get("width"), record.get("height")) != (420, 380):
            raise ValueError(f"Kích thước metadata sai: {name}")
    geometry = metadata.get("stereo_geometry", {})
    required_geometry = {"left": "frame10.png", "right": "frame11.png",
                         "source_image_size": [434, 383], "left_crop_xy": [10, 0], "right_crop_xy": [0, 0]}
    if any(geometry.get(key) != value for key, value in required_geometry.items()):
        raise ValueError("Metadata hình học không khớp crop Venus đã xác minh.")
    images = [cv2.imread(str(input_dir / name), cv2.IMREAD_COLOR) for name in ("frame10.png", "frame11.png")]
    if any(image is None or image.shape != (380, 420, 3) for image in images):
        raise ValueError("Ảnh Venus không giải mã được hoặc sai kích thước.")
    ground_truth = flow_ground_truth(read_flow(input_dir / "flow10.flo", (420, 380)))
    return images[0], images[1], ground_truth, metadata


def assumed_camera_model() -> dict:
    """Mô hình pinhole theo FoV 30° của Basha et al. (2013), §4.1.

    Dự án chọn FoV ngang, baseline=1 và principal point ở tâm ảnh nguồn.
    Crop trái (10,0), phải (0,0) chuyển tọa độ từ im2/im6 sang frame10/frame11.
    """
    focal = 434 / (2 * math.tan(math.radians(30) / 2))
    left_intrinsic = np.array([[focal, 0, 206.5], [0, focal, 191], [0, 0, 1]])
    right_intrinsic = np.array([[focal, 0, 216.5], [0, focal, 191], [0, 0, 1]])
    left_projection = left_intrinsic @ np.column_stack((np.eye(3), np.zeros(3)))
    right_projection = right_intrinsic @ np.column_stack((np.eye(3), [-1, 0, 0]))
    reprojection = [[1, 0, 0, -206.5], [0, 1, 0, -191], [0, 0, 0, focal], [0, 0, 1, 10]]
    return {
        "camera_model_kind": "assumed", "length_unit": "normalized_baseline_units",
        "coordinate_frame": "Left camera: X right, Y down, Z forward.",
        "label": "Assumed camera model; normalized units", "reference": "Basha2013, section 4.1",
        "image_size": [420, 380], "source_image_size": [434, 383],
        "horizontal_fov_degrees": 30, "focal_px": focal, "baseline": 1.0,
        "crop_xy_left": [10, 0], "crop_xy_right": [0, 0], "doffs_px": 10.0,
        "K_left": left_intrinsic.tolist(), "K_right": right_intrinsic.tolist(),
        "P_left": left_projection.tolist(), "P_right": right_projection.tolist(), "Q": reprojection,
        "additional_assumptions": ["30 degrees interpreted as horizontal FoV on the original 434x383 images",
            "square pixels, zero skew, parallel cameras, baseline normalized to 1",
            "principal points at the source image center; no original principal-point difference",
            "zero distortion in the already rectified input coordinate frame"],
        "crop_evidence": {"source_archive": "https://vision.middlebury.edu/stereo/data/scenes2001/data/venus/venus.zip",
            "left_source": "im2.ppm", "right_source": "im6.ppm", "verification": "Exact RGB pixel equality of the 420x380 crops, checked during dataset preparation.",
            "left_source_sha256": "fef870c1e2471fe5b3b2bac9ca71810e8503e1489e4203288a982b40355b9172",
            "right_source_sha256": "b2fca7b6f3aa7e1a34bc5eac2d99cbbc5dfc77ce531777b71153e2ed73290004"},
        "input_sha256": {name: INPUT_HASHES[name] for name in ("frame10.png", "frame11.png")},
        "depth_unit_description": "Coordinates are expressed in units of the selected baseline B=1.",
    }


def extract_correspondences(left: np.ndarray, right: np.ndarray, config: dict) -> dict:
    detector = cv2.SIFT_create(nfeatures=config["max_features"])
    keys_left, descriptors_left = detector.detectAndCompute(left, None)
    keys_right, descriptors_right = detector.detectAndCompute(right, None)
    if descriptors_left is None or descriptors_right is None or len(keys_right) < 2:
        raise ValueError("Không đủ descriptor để ghép KNN.")
    def records(keypoints):
        return np.array([[*key.pt, key.size, key.angle, key.response, key.octave, key.class_id] for key in keypoints], np.float64)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_left, descriptors_right, k=2)
    rows = []
    for neighbours in knn:
        if len(neighbours) == 2:
            first, second = neighbours
            rows.append([first.queryIdx, first.trainIdx, second.trainIdx, first.distance, second.distance])
    candidates = np.asarray(rows, np.float64).reshape(-1, 5)
    accepted = candidates[:, 3] < config["ratio"] * candidates[:, 4]
    indices = candidates[accepted, :2].astype(np.int32)
    left_records, right_records = records(keys_left), records(keys_right)
    return {"features_left": left_records, "features_right": right_records,
            "descriptors_left": descriptors_left, "descriptors_right": descriptors_right,
            "knn": candidates, "ratio_mask": accepted, "match_indices": indices,
            "points_left": left_records[indices[:, 0], :2].astype(np.float32),
            "points_right": right_records[indices[:, 1], :2].astype(np.float32)}


def epipolar_errors(matrix: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict:
    if matrix.shape != (3, 3) or left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2:
        raise ValueError("Sai shape F hoặc correspondence.")
    if not all(np.isfinite(value).all() for value in (matrix, left, right)):
        raise ValueError("F/correspondence có giá trị không hữu hạn.")
    left_h = np.column_stack((left, np.ones(len(left)))).astype(np.float64)
    right_h = np.column_stack((right, np.ones(len(right)))).astype(np.float64)
    lines_right = left_h @ matrix.T
    lines_left = right_h @ matrix
    residual = np.sum(right_h * lines_right, axis=1)
    right_norm = np.sum(lines_right[:, :2] ** 2, axis=1)
    left_norm = np.sum(lines_left[:, :2] ** 2, axis=1)
    defined = (left_norm > 1e-20) & (right_norm > 1e-20)
    symmetric = np.full(len(left), np.nan)
    sampson_sq = np.full(len(left), np.nan)
    symmetric[defined] = 0.5 * np.abs(residual[defined]) * (1 / np.sqrt(left_norm[defined]) + 1 / np.sqrt(right_norm[defined]))
    sampson_sq[defined] = residual[defined] ** 2 / (left_norm[defined] + right_norm[defined])
    return {"epipolar_px": symmetric, "sampson_px": np.sqrt(sampson_sq), "sampson_sq_px2": sampson_sq}


def estimate_fundamental(left: np.ndarray, right: np.ndarray, threshold: float, config: dict) -> dict:
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2 or len(left) < 8:
        raise ValueError("Cần ít nhất tám correspondence dạng N×2.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Correspondence không hữu hạn.")
    if len(np.unique(left, axis=0)) < 8 or len(np.unique(right, axis=0)) < 8:
        raise ValueError("Correspondence trùng lặp/suy biến.")
    cv2.setRNGSeed(config["seed"])
    start = time.perf_counter_ns()
    # Preset USAC có sigma-consensus và random state riêng bằng 0, không dùng RNG toàn cục.
    matrix, mask = cv2.findFundamentalMat(left, right, cv2.USAC_MAGSAC, threshold, config["confidence"], config["max_iters"])
    elapsed = (time.perf_counter_ns() - start) / 1e6
    if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all() or mask is None:
        raise ValueError("RANSAC không tìm được F hữu hạn 3×3.")
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[0] <= 0 or singular[1] <= singular[0] * 1e-6 or singular[2] > singular[0] * 1e-6:
        raise ValueError("Fundamental matrix không có rank 2 ổn định.")
    matrix = matrix / np.linalg.norm(matrix)
    mask = mask.ravel().astype(bool)
    if mask.shape != (len(left),) or mask.sum() < 8:
        raise ValueError("RANSAC có ít hơn tám inlier.")
    return {"F": matrix, "inliers": mask, **epipolar_errors(matrix, left, right), "ransac_ms": elapsed}


def geometry_metrics(result: dict, left: np.ndarray, right: np.ndarray) -> dict:
    mask = result["inliers"]
    metrics = {"ratio_pass_matches": len(left), "inliers": int(mask.sum()),
               "inlier_ratio_pct": 100 * float(mask.mean()),
               "inlier_vertical_median_px": float(np.median(np.abs(left[mask, 1] - right[mask, 1]))),
               "inlier_vertical_p95_px": float(np.percentile(np.abs(left[mask, 1] - right[mask, 1]), 95))}
    for name in ("epipolar_px", "sampson_px", "sampson_sq_px2"):
        for label, selected in (("all", np.ones(len(mask), bool)), ("inlier", mask)):
            values = result[name][selected]
            finite = np.isfinite(values)
            metrics[f"{label}_mean_{name}"] = float(values[finite].mean()) if finite.any() else None
            metrics[f"{label}_undefined_{name}"] = int((~finite).sum())
    return metrics


def geometry_diagnostics(matrix: np.ndarray, ground_truth: dict) -> dict:
    """Đánh giá F bằng GT độc lập; không đưa GT vào bước ước lượng."""
    y, x = np.nonzero(ground_truth["evaluation_domain"])
    if not len(x):
        return {"gt_domain_correspondences": 0, "geometry_quality": "unavailable",
                "quality_note": "No GT-valid/in-bounds correspondences."}
    left = np.column_stack((x, y)).astype(np.float64)
    right = left.copy()
    right[:, 0] -= ground_truth["disparity"][y, x]
    errors = epipolar_errors(matrix, left, right)
    metrics = {"gt_domain_correspondences": len(x)}
    for name in ("epipolar_px", "sampson_px"):
        values = errors[name]
        finite = np.isfinite(values)
        metrics["gt_domain_undefined_" + name] = int((~finite).sum())
        metrics["gt_domain_mean_" + name] = float(values[finite].mean()) if finite.any() else None
        metrics["gt_domain_p95_" + name] = float(np.percentile(values[finite], 95)) if finite.any() else None
    for side, points, transform in (("left", right, matrix), ("right", left, matrix.T)):
        lines = np.column_stack((points, np.ones(len(points)))) @ transform
        defined = np.linalg.norm(lines[:, :2], axis=1) > 1e-10
        angles = (np.degrees(np.arctan2(-lines[defined, 0], lines[defined, 1])) + 90) % 180 - 90
        metrics[f"{side}_undefined_lines"] = int((~defined).sum())
        metrics[f"{side}_angle_median_deg"] = float(np.median(np.abs(angles))) if len(angles) else None
        metrics[f"{side}_angle_p95_deg"] = float(np.percentile(np.abs(angles), 95)) if len(angles) else None
    # Ngưỡng chẩn đoán do dự án chọn, không phải ngưỡng chấm điểm của đề.
    limit = GEOMETRY_PROTOCOL["diagnostic_angle_p95_limit_deg"]
    stable = all(metrics[f"{side}_undefined_lines"] == 0 and metrics[f"{side}_angle_p95_deg"] <= limit for side in ("left", "right"))
    metrics.update(geometry_quality="ok" if stable else "warning",
                   quality_note="Rectified-pair diagnostic: p95 absolute epipolar angle <= 2 degrees on both views.")
    return metrics


def create_matcher(method: str, num_disparities: int, block_size: int, min_disparity: int):
    if any(type(value) is not int for value in (num_disparities, block_size, min_disparity)):
        raise ValueError("Tham số matcher phải là số nguyên.")
    if num_disparities <= 0 or num_disparities % 16 or block_size < 5 or block_size > 255 or block_size % 2 != 1:
        raise ValueError("numDisparities phải chia hết cho 16; blockSize lẻ từ 5 đến 255.")
    if method == "bm":
        matcher = cv2.StereoBM_create(num_disparities, block_size)
        matcher.setMinDisparity(min_disparity)
        matcher.setTextureThreshold(10)
        matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
        matcher.setPreFilterSize(9)
    elif method == "sgbm":
        matcher = cv2.StereoSGBM_create(min_disparity, num_disparities, block_size,
            P1=8 * block_size ** 2, P2=32 * block_size ** 2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    else:
        raise ValueError(f"Phương pháp không được hỗ trợ: {method}")
    matcher.setUniquenessRatio(10)
    matcher.setPreFilterCap(31)
    matcher.setSpeckleWindowSize(0)
    matcher.setSpeckleRange(0)
    # SGBM 5.0 đổi giá trị không dương thành 1; ngưỡng này vượt mọi sai lệch trong grid.
    matcher.setDisp12MaxDiff(1_000_000)
    return matcher


def matcher_settings(matcher, method: str) -> dict:
    names = ["MinDisparity", "NumDisparities", "BlockSize", "UniquenessRatio", "PreFilterCap", "SpeckleWindowSize", "SpeckleRange", "Disp12MaxDiff"]
    names += ["TextureThreshold", "PreFilterType", "PreFilterSize"] if method == "bm" else ["P1", "P2", "Mode"]
    return {name[0].lower() + name[1:]: getattr(matcher, "get" + name)() for name in names}


def disparity_masks(raw_left: np.ndarray, raw_right: np.ndarray, min_left: int, min_right: int, num_disparities: int, tolerance: float) -> dict:
    if raw_left.shape != raw_right.shape or raw_left.ndim != 2 or raw_left.dtype != np.int16 or raw_right.dtype != np.int16:
        raise ValueError("Disparity native phải cùng shape H×W và dtype int16.")
    left = raw_left.astype(np.float32) / 16
    right = raw_right.astype(np.float32) / 16
    valid_left = (raw_left > (min_left - 1) * 16) & (left >= min_left) & (left < min_left + num_disparities)
    valid_right = (raw_right > (min_right - 1) * 16) & (right >= min_right) & (right < min_right + num_disparities)
    height, width = left.shape
    y, x = np.indices((height, width))
    correspondence_x = x - left
    in_bounds = valid_left & (correspondence_x >= 0) & (correspondence_x <= width - 1)
    nearest_x = np.clip(np.rint(correspondence_x).astype(np.int32), 0, width - 1)
    reverse_valid = valid_right[y, nearest_x]
    errors = np.abs(left + right[y, nearest_x])
    comparable = in_bounds & reverse_valid
    errors[~comparable] = np.nan
    return {"disparity": left, "disparity_right": right, "raw_valid": valid_left,
            "raw_valid_right": valid_right, "correspondence_in_bounds": in_bounds,
            "lr_valid": comparable & (errors <= tolerance), "lr_error_px": errors}


def disparity_metrics(disparity: np.ndarray, masks: dict, ground_truth: dict, bad_threshold: float) -> dict:
    reference = ground_truth["disparity"]
    domain = ground_truth["evaluation_domain"]
    if disparity.shape != reference.shape or domain.shape != disparity.shape:
        raise ValueError("Prediction và GT phải cùng hệ tọa độ/kích thước.")
    count = int(domain.sum())
    result = {"raw_valid_ratio_pct": 100 * float(masks["raw_valid"].mean()),
              "consistent_valid_ratio_pct": 100 * float(masks["lr_valid"].mean()),
              "gt_valid_pixels": int(ground_truth["valid"].sum()), "gt_domain_pixels": count,
              "bad_pixel_threshold_px": bad_threshold}
    errors = np.abs(disparity.astype(np.float64) - reference)
    for prefix, valid in (("raw_", masks["raw_valid"]), ("", masks["lr_valid"])):
        selected = domain & valid
        values = errors[selected]
        if not np.isfinite(values).all():
            raise ValueError("Prediction được đánh dấu hợp lệ nhưng chứa NaN/Inf.")
        result[prefix + "evaluated_pixels"] = int(selected.sum())
        result[prefix + "coverage_pct"] = 100 * len(values) / count if count else None
        result[prefix + "rmse_px"] = float(np.sqrt(np.mean(values ** 2))) if len(values) else None
        result[prefix + "bad_pixel_pct"] = 100 * float(np.mean(values > bad_threshold)) if len(values) else None
    bad_or_invalid = domain & (~masks["lr_valid"] | (errors > bad_threshold))
    result["bad_or_invalid_pct"] = 100 * int(bad_or_invalid.sum()) / count if count else None
    result["evaluation_note"] = "LR-consistent prediction intersected with GT-valid/in-bounds domain; no occlusion GT mask."
    if count == 0:
        result["evaluation_note"] += " No GT-valid/in-bounds pixels; scores and coverage are N/A."
    elif result["evaluated_pixels"] == 0:
        result["evaluation_note"] += " No LR-consistent prediction in the GT domain; conditional scores are N/A."
    return result


def common_domain_comparison(rows: list[dict], case_root: Path, ground_truth: dict, bad_threshold: float) -> list[dict]:
    """So BM/SGBM cùng cấu hình trên đúng giao mask; giữ cả cặp thất bại."""
    comparison = []
    for row in rows:
        if row["method"] != "bm":
            continue
        partner = next(item for item in rows if item["method"] == "sgbm" and
                       item["num_disparities"] == row["num_disparities"] and item["block_size"] == row["block_size"])
        pair = (row, partner)
        if any(item["status"] != "ok" for item in pair):
            for item in pair:
                comparison.append({"case_id": item["case_id"], "status": "unavailable", "reason": "One or both dense cases failed."})
            continue
        arrays = [_load_arrays(case_root / item["case_id"] / "disparity.npz") for item in pair]
        common = ground_truth["evaluation_domain"] & arrays[0]["lr_valid"] & arrays[1]["lr_valid"]
        for item, prediction in zip(pair, arrays):
            values = disparity_metrics(prediction["disparity"], {"raw_valid": common, "lr_valid": common}, ground_truth, bad_threshold)
            comparison.append({"case_id": item["case_id"], "status": "ok" if common.any() else "unavailable",
                "common_pixels": values["evaluated_pixels"], "common_coverage_pct": values["coverage_pct"],
                "common_rmse_px": values["rmse_px"], "common_bad_pixel_pct": values["bad_pixel_pct"],
                "reason": "Same BM/SGBM LR-valid intersection with GT domain." if common.any() else "Empty common domain; scores are N/A."})
    return comparison


def reproject(disparity: np.ndarray, valid: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    if disparity.ndim != 2 or valid.shape != disparity.shape or valid.dtype != bool:
        raise ValueError("Disparity/mask reprojection không hợp lệ.")
    if list(disparity.shape[::-1]) != camera["image_size"]:
        raise ValueError("Camera không khớp kích thước ảnh.")
    matrix = np.asarray(camera["Q"], np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 4:
        raise ValueError("Q phải hữu hạn, không suy biến và có shape 4×4.")
    y, x = np.indices(disparity.shape)
    denominator = matrix[3, 0] * x + matrix[3, 1] * y + matrix[3, 2] * disparity + matrix[3, 3]
    denominator_ok = np.isfinite(disparity) & np.isfinite(denominator) & (denominator > 1e-8)
    safe_disparity = np.where(np.isfinite(disparity), disparity, 0).astype(np.float32)
    xyz = cv2.reprojectImageTo3D(safe_disparity, matrix, handleMissingValues=False)
    finite = np.isfinite(xyz).all(axis=2)
    positive = xyz[:, :, 2] > 0
    support = valid & denominator_ok & finite & positive
    statistics = {"candidate_pixels": int(valid.sum()),
        "rejected_denominator": int((valid & ~denominator_ok).sum()),
        "rejected_nonfinite": int((valid & denominator_ok & ~finite).sum()),
        "rejected_depth": int((valid & denominator_ok & finite & ~positive).sum()),
        "point_count": int(support.sum())}
    xyz[~support] = np.nan
    return xyz, support, statistics


def filter_cloud_support(disparity: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, dict]:
    """Loại cụm disparity nhỏ độc lập GT; không sửa disparity hoặc LR mask."""
    if disparity.shape != support.shape or disparity.ndim != 2 or support.dtype != bool:
        raise ValueError("Disparity/support lọc cloud phải cùng shape H×W.")
    values = disparity[support]
    if not np.isfinite(values).all() or np.any(values < -2047) or np.any(values > 2047):
        raise ValueError("Disparity cloud không hữu hạn hoặc vượt fixed-point int16.")
    sentinel = int(np.iinfo(np.int16).min)
    native = np.full(disparity.shape, sentinel, np.int16)
    native[support] = np.rint(values * 16).astype(np.int16)
    # maxDiff cũng ở đơn vị fixed-point: 16 tương ứng 1 px.
    cv2.filterSpeckles(native, sentinel, 100, 16)
    filtered = support & (native != sentinel)
    return filtered, {"filter_method": "disparity_connected_components", "max_component_pixels": 100,
        "max_neighbor_disparity_difference_px": 1.0, "uses_ground_truth": False,
        "input_points": int(support.sum()), "removed_points": int((support & ~filtered).sum()),
        "retained_points": int(filtered.sum())}


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape != colors.shape or len(points) == 0:
        raise ValueError("PLY cần điểm và RGB dạng N×3, N>0.")
    if not np.isfinite(points).all() or colors.dtype != np.uint8:
        raise ValueError("PLY không nhận điểm không hữu hạn hoặc màu ngoài uint8.")
    with Path(path).open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\ncomment Assumed camera model; normalized units\n")
        stream.write("comment Basha et al. 2013 section 4.1 DOI 10.1007/s11263-012-0542-7\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(stream, np.column_stack((points, colors)), fmt=["%.9g"] * 3 + ["%d"] * 3)


def _validate_config(config: dict) -> None:
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError("Thiếu hoặc thừa khóa cấu hình Part B.")
    if config["methods"] != ["bm", "sgbm"]:
        raise ValueError("So sánh Part B phải gồm BM và SGBM theo thứ tự này.")
    for name in ("num_disparities", "block_sizes", "ransac_thresholds_px"):
        values = config[name]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"Grid {name} phải khác rỗng và không trùng.")
    for name in ("warmup", "repetitions", "max_features", "max_iters", "seed", "min_disparity"):
        if type(config[name]) is not int:
            raise ValueError(f"{name} phải là số nguyên.")
    if config["warmup"] < 0 or config["repetitions"] < 1 or config["max_features"] < 8 or config["max_iters"] < 1:
        raise ValueError("Số lần đo/features/iterations không hợp lệ.")
    if not 0 <= config["seed"] < 2 ** 31:
        raise ValueError("Seed phải nằm trong phạm vi int32 không âm.")
    for name in ("confidence", "ratio", "lr_tolerance_px", "bad_pixel_threshold_px"):
        if isinstance(config[name], bool) or not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} phải hữu hạn và dương.")
    if config["confidence"] >= 1 or config["ratio"] >= 1:
        raise ValueError("Confidence và Lowe ratio phải nhỏ hơn 1.")
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in config["ransac_thresholds_px"]):
        raise ValueError("Ngưỡng RANSAC phải hữu hạn và dương.")
    if 1.0 not in config["ransac_thresholds_px"] or 9 not in config["block_sizes"] or 32 not in config["num_disparities"]:
        raise ValueError("Grid cần chứa cấu hình tham chiếu: threshold=1, blockSize=9, numDisparities=32.")
    for number in config["num_disparities"]:
        if config["min_disparity"] < -2047 or config["min_disparity"] + number > 2047:
            raise ValueError("Khoảng disparity vượt giới hạn fixed-point int16.")
        for block in config["block_sizes"]:
            create_matcher("bm", number, block, config["min_disparity"])
            if max(config["min_disparity"] + number, 0) + max(-config["min_disparity"], 0) + block >= 420:
                raise ValueError("Ảnh quá nhỏ cho khoảng disparity và blockSize.")


def _cases(config: dict) -> list[dict]:
    cases = []
    for method in config["methods"]:
        for number in config["num_disparities"]:
            for block in config["block_sizes"]:
                cases.append({"case_id": f"{method}_n{number}_b{block}", "method": method,
                    "num_disparities": number, "block_size": block, "min_disparity": config["min_disparity"]})
    return cases


def _geometry_id(threshold: float) -> str:
    return "ransac_t" + f"{threshold:g}".replace(".", "p")


def _environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "opencv": cv2.__version__, "opencv_threads": 1, "opencl": False,
            "packages": {name: version(name) for name in ("numpy", "matplotlib", "opencv-contrib-python-headless")}}


def _compute_case(left: np.ndarray, right: np.ndarray, case: dict, config: dict) -> tuple[dict, list[dict], dict]:
    timings = []
    previous = None
    deterministic = True
    for index in range(-config["warmup"], config["repetitions"]):
        start = time.perf_counter_ns()
        matcher_left = create_matcher(case["method"], case["num_disparities"], case["block_size"], case["min_disparity"])
        min_right = -case["min_disparity"] - case["num_disparities"] + 1
        matcher_right = create_matcher(case["method"], case["num_disparities"], case["block_size"], min_right)
        after_creation = time.perf_counter_ns()
        raw_left = matcher_left.compute(left, right)
        after_left = time.perf_counter_ns()
        raw_right = matcher_right.compute(right, left)
        after_right = time.perf_counter_ns()
        masks = disparity_masks(raw_left, raw_right, case["min_disparity"], min_right, case["num_disparities"], config["lr_tolerance_px"])
        finish = time.perf_counter_ns()
        if index >= 0:
            if previous is not None:
                deterministic &= np.array_equal(previous[0], raw_left) and np.array_equal(previous[1], raw_right)
            previous = (raw_left.copy(), raw_right.copy())
            timings.append({"case_id": case["case_id"], "repeat": index,
                "matcher_creation_ms": (after_creation - start) / 1e6,
                "left_matching_ms": (after_left - after_creation) / 1e6,
                "right_matching_ms": (after_right - after_left) / 1e6,
                "consistency_ms": (finish - after_right) / 1e6,
                "stereo_pipeline_ms": (finish - start) / 1e6})
    settings = {"left": matcher_settings(matcher_left, case["method"]),
                "right": matcher_settings(matcher_right, case["method"]), "repeat_deterministic": bool(deterministic)}
    return {"raw_left": raw_left, "raw_right": raw_right, **masks}, timings, settings


def _inventory(run_dir: Path) -> dict:
    return {str(path.relative_to(run_dir)): _digest(path) for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.name not in ("run.json", "validation.json")}


def timing_summary(timings: list[dict]) -> dict:
    stages = ("matcher_creation_ms", "left_matching_ms", "right_matching_ms", "consistency_ms", "stereo_pipeline_ms")
    summary = {name: float(np.median([float(entry[name]) for entry in timings])) for name in stages}
    values = np.array([float(entry["stereo_pipeline_ms"]) for entry in timings])
    summary.update(timing_repetitions=len(timings), stereo_pipeline_min_ms=float(values.min()),
        stereo_pipeline_max_ms=float(values.max()), stereo_pipeline_std_ms=float(values.std()),
        stereo_pipeline_cv_pct=float(100 * values.std() / values.mean()) if values.mean() > 0 else None)
    return summary


def run_experiments(input_dir: Path = DEFAULT_INPUT, output_root: Path = DEFAULT_OUTPUT, config: dict | None = None) -> Path:
    config = copy.deepcopy(DEFAULT_CONFIG if config is None else config)
    _validate_config(config)
    left, right, ground_truth, metadata = load_stereo_pair(input_dir)
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    run_dir = Path(output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    inputs = run_dir / "inputs"
    inputs.mkdir()
    for name in [*INPUT_HASHES, "metadata.json"]:
        shutil.copyfile(Path(input_dir) / name, inputs / name)
    source = Path(__file__).read_bytes()
    (run_dir / "source_code.py").write_bytes(source)
    camera = assumed_camera_model()
    _write_json(run_dir / "camera.json", camera)
    _write_json(run_dir / "references.json", REFERENCES)
    np.savez_compressed(run_dir / "ground_truth.npz", **ground_truth)
    manifest = {
        "status": "running", "config": config, "environment": _environment(),
        "source_sha256": hashlib.sha256(source).hexdigest(), "input_metadata": metadata,
        "camera_model_kind": "assumed", "physical_3d_accuracy_established": False,
        "disparity_convention": "x_left - x_right in current cropped images; GT = -flow_u",
        "rectification": {"evidence": "All valid GT flow has |v| <= 0.0001 px; no additional warp.",
                          "occlusion_ground_truth_available": False},
        "geometry_protocol": copy.deepcopy(GEOMETRY_PROTOCOL),
        "timing_protocol": {"unit": "ms", "warmup": config["warmup"], "repetitions": config["repetitions"],
            "dense_scope": "matcher creation, left/right native computation, conversion and consistency masks; excludes I/O, evaluation, figures and 3D",
            "geometry_scope": "SIFT plus KNN/ratio and RANSAC timed separately as single samples"},
        "point_clouds": {}, "errors": [], "warnings": [],
    }
    _write_json(run_dir / "run.json", manifest)
    previous_threads, previous_opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    geometry_rows, rows, timings = [], [], []
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        cv2.setRNGSeed(config["seed"])
        geometry_dir = run_dir / "geometry"
        geometry_dir.mkdir()
        matches = None
        try:
            start = time.perf_counter_ns()
            matches = extract_correspondences(gray_left, gray_right, config)
            manifest["feature_matching_ms"] = (time.perf_counter_ns() - start) / 1e6
            np.savez_compressed(geometry_dir / "matches.npz", **matches)
        except (ValueError, cv2.error) as error:
            manifest["errors"].append(f"features: {error}")
        for threshold in config["ransac_thresholds_px"]:
            row = {"geometry_id": _geometry_id(threshold), "threshold_px": threshold, "status": "failed", "error": ""}
            try:
                if matches is None:
                    raise ValueError("Không có correspondences từ bước feature matching.")
                result = estimate_fundamental(matches["points_left"], matches["points_right"], threshold, config)
                np.savez_compressed(geometry_dir / f"{row['geometry_id']}.npz", **result)
                row.update(geometry_metrics(result, matches["points_left"], matches["points_right"]))
                row.update(geometry_diagnostics(result["F"], ground_truth))
                if row["geometry_quality"] != "ok":
                    manifest["warnings"].append(f"{row['geometry_id']}: {row['geometry_quality']}; {row['quality_note']}")
                row.update(status="ok", ransac_ms=result["ransac_ms"],
                    detected_left=len(matches["features_left"]), detected_right=len(matches["features_right"]),
                    initial_matches=len(matches["knn"]))
            except (ValueError, cv2.error) as error:
                row["error"] = str(error)
            geometry_rows.append(row)
        case_root = run_dir / "cases"
        case_root.mkdir()
        for case in _cases(config):
            case_dir = case_root / case["case_id"]
            case_dir.mkdir()
            row = {**case, "status": "failed", "error": "", "stage": "dense_matching"}
            try:
                arrays, case_timings, settings = _compute_case(gray_left, gray_right, case, config)
                np.savez_compressed(case_dir / "disparity.npz", **arrays)
                _write_json(case_dir / "settings.json", settings)
                row.update(disparity_metrics(arrays["disparity"], arrays, ground_truth, config["bad_pixel_threshold_px"]))
                row.update(timing_summary(case_timings))
                row.update(status="ok", stage="complete", repeat_deterministic=settings["repeat_deterministic"])
                if not settings["repeat_deterministic"]:
                    manifest["warnings"].append(f"{case['case_id']}: raw disparity changed between timing repetitions; arrays are from the final repetition.")
                timings.extend(case_timings)
                if case["num_disparities"] == 32 and case["block_size"] == 9:
                    try:
                        xyz, support, statistics = reproject(arrays["disparity"], arrays["lr_valid"], camera)
                        colors = left[:, :, ::-1][support]
                        points = xyz[support]
                        y, x = np.nonzero(support)
                        write_ply(case_dir / "cloud.ply", points, colors)
                        np.savez_compressed(case_dir / "cloud.npz", points=points, colors=colors,
                            pixels_xy=np.column_stack((x, y)).astype(np.int32), depth=xyz[:, :, 2], support=support)
                        filtered, filter_statistics = filter_cloud_support(arrays["disparity"], support)
                        write_ply(case_dir / "cloud_filtered.ply", xyz[filtered], left[:, :, ::-1][filtered])
                        filtered_y, filtered_x = np.nonzero(filtered)
                        np.savez_compressed(case_dir / "cloud_filtered.npz", points=xyz[filtered], colors=left[:, :, ::-1][filtered],
                            pixels_xy=np.column_stack((filtered_x, filtered_y)).astype(np.int32),
                            depth=np.where(filtered, xyz[:, :, 2], np.nan), support=filtered)
                        manifest["point_clouds"][case["case_id"]] = {"status": "ok", **statistics,
                            "camera_model_kind": "assumed", "unit": camera["length_unit"], "filter": filter_statistics}
                    except (ValueError, cv2.error) as error:
                        manifest["point_clouds"][case["case_id"]] = {"status": "failed", "error": str(error)}
            except (ValueError, cv2.error) as error:
                row["error"] = str(error)
            rows.append(row)
            _write_json(case_dir / "case.json", row)
        _write_csv(run_dir / "geometry_metrics.csv", geometry_rows)
        _write_csv(run_dir / "metrics.csv", rows)
        _write_csv(run_dir / "timings.csv", timings)
        _write_csv(run_dir / "common_domain_metrics.csv", common_domain_comparison(rows, case_root, ground_truth, config["bad_pixel_threshold_px"]))
        all_ok = (all(row["status"] == "ok" for row in [*rows, *geometry_rows]) and
                  len(manifest["point_clouds"]) == 2 and all(cloud["status"] == "ok" for cloud in manifest["point_clouds"].values()))
        manifest["status"] = "completed_with_assumed_camera" if all_ok else "partial_failed_cases"
        _write_json(run_dir / "run.json", manifest)
        build_figures(run_dir)
        validate_run(run_dir)
    except BaseException as error:
        manifest = _read_json(run_dir / "run.json")
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
        _write_json(run_dir / "run.json", manifest)
        raise
    finally:
        cv2.setNumThreads(previous_threads)
        cv2.ocl.setUseOpenCL(previous_opencl)
    return run_dir


def find_matching_run(input_dir: Path = DEFAULT_INPUT, output_root: Path = DEFAULT_OUTPUT, config: dict | None = None) -> Path:
    config = DEFAULT_CONFIG if config is None else config
    _validate_config(config)
    _, _, _, metadata = load_stereo_pair(input_dir)
    for run_dir in sorted(Path(output_root).glob("*"), reverse=True):
        try:
            manifest = _read_json(run_dir / "run.json")
            if (manifest["status"] == "completed_with_assumed_camera" and manifest["config"] == config and
                manifest["input_metadata"] == metadata and manifest["source_sha256"] == _digest(Path(__file__)) and
                manifest["environment"] == _environment()):
                validate_run(run_dir)
                return run_dir
        except (OSError, ValueError, KeyError):
            continue
    raise FileNotFoundError("Không tìm thấy full run tương thích. Chạy CLI hoặc bật rerun_pipeline trong notebook.")


def _check_row(recorded: dict, expected: dict) -> None:
    for key, value in expected.items():
        actual = recorded.get(key)
        if value is None:
            if actual not in (None, ""):
                raise ValueError(f"Metric {key} phải là N/A.")
        elif isinstance(value, bool):
            if type(actual) is not bool and actual not in ("True", "False"):
                raise ValueError(f"Trường {key} không phải boolean.")
            if (actual is True or actual == "True") != value:
                raise ValueError(f"Trường {key} không khớp.")
        elif isinstance(value, (int, float, np.number)):
            if actual in (None, "") or not math.isclose(float(actual), float(value), rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(f"Metric {key} không khớp arrays: {actual} != {value}")
        elif actual != value:
            raise ValueError(f"Trường {key} không khớp.")


def _load_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def validate_run(run_dir: Path) -> dict:
    """Đọc lại raw arrays để kiểm chứng masks, metrics, reprojection và PLY."""
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "run.json")
    config = manifest["config"]
    _validate_config(config)
    _check_row(manifest, {"camera_model_kind": "assumed", "physical_3d_accuracy_established": False})
    if manifest.get("geometry_protocol") != GEOMETRY_PROTOCOL:
        raise ValueError("Run không dùng geometry protocol hiện tại; kiểm tra run cũ bằng source snapshot đi kèm.")
    actual_inventory = _inventory(run_dir)
    if actual_inventory != manifest.get("artifacts"):
        raise ValueError("Danh sách/checksum artifacts đã thay đổi.")
    if _digest(run_dir / "source_code.py") != manifest["source_sha256"]:
        raise ValueError("Source snapshot không khớp hash.")
    left, _, ground_truth, metadata = load_stereo_pair(run_dir / "inputs")
    if metadata != manifest["input_metadata"]:
        raise ValueError("Input metadata không khớp snapshot.")
    saved_gt = _load_arrays(run_dir / "ground_truth.npz")
    for key, value in ground_truth.items():
        if not np.array_equal(value, saved_gt[key], equal_nan=True):
            raise ValueError(f"GT đã đổi: {key}")
    camera = _read_json(run_dir / "camera.json")
    if camera != assumed_camera_model() or _read_json(run_dir / "references.json") != REFERENCES:
        raise ValueError("Camera/references không khớp mô hình giả định đã khai báo.")
    rows = read_csv(run_dir / "metrics.csv")
    cases = _cases(config)
    if [row["case_id"] for row in rows] != [case["case_id"] for case in cases]:
        raise ValueError("Bảng dense thiếu/thừa hoặc sai thứ tự cấu hình.")
    timings = read_csv(run_dir / "timings.csv")
    if any(entry["case_id"] not in {case["case_id"] for case in cases} for entry in timings):
        raise ValueError("Timing tham chiếu cấu hình không tồn tại.")
    expected_cloud_ids = {f"{method}_n32_b9" for method in config["methods"]}
    if set(manifest["point_clouds"]) - expected_cloud_ids:
        raise ValueError("Point cloud không thuộc cấu hình tham chiếu.")
    successful = 0
    for case, row in zip(cases, rows):
        directory = run_dir / "cases" / case["case_id"]
        _check_row(row, case)
        saved_case = _read_json(directory / "case.json")
        _check_row(row, saved_case)
        if row["status"] != "ok":
            if not row["error"]:
                raise ValueError("Cấu hình thất bại không có lý do.")
            continue
        arrays = _load_arrays(directory / "disparity.npz")
        min_right = -case["min_disparity"] - case["num_disparities"] + 1
        recomputed = disparity_masks(arrays["raw_left"], arrays["raw_right"], case["min_disparity"], min_right, case["num_disparities"], config["lr_tolerance_px"])
        for key, value in recomputed.items():
            if not np.array_equal(value, arrays[key], equal_nan=True):
                raise ValueError(f"Mask/disparity không khớp: {case['case_id']} {key}")
        _check_row(row, disparity_metrics(arrays["disparity"], arrays, ground_truth, config["bad_pixel_threshold_px"]))
        settings = _read_json(directory / "settings.json")
        _check_row(row, {"repeat_deterministic": settings["repeat_deterministic"]})
        for side, minimum in (("left", case["min_disparity"]), ("right", min_right)):
            expected = matcher_settings(create_matcher(case["method"], case["num_disparities"], case["block_size"], minimum), case["method"])
            if settings[side] != expected:
                raise ValueError("Tham số matcher không khớp.")
        case_timings = [entry for entry in timings if entry["case_id"] == case["case_id"]]
        if [int(entry["repeat"]) for entry in case_timings] != list(range(config["repetitions"])):
            raise ValueError("Thiếu/thừa repetition timing.")
        stage_names = ("matcher_creation_ms", "left_matching_ms", "right_matching_ms", "consistency_ms")
        for entry in case_timings:
            values = [float(entry[key]) for key in (*stage_names, "stereo_pipeline_ms")]
            if any(not math.isfinite(value) or value < 0 for value in values) or not math.isclose(sum(values[:-1]), values[-1], abs_tol=1e-6):
                raise ValueError("Timing không hữu hạn hoặc sai phạm vi tổng.")
        _check_row(row, timing_summary(case_timings))
        cloud = manifest["point_clouds"].get(case["case_id"])
        if cloud and cloud["status"] == "ok":
            xyz, support, statistics = reproject(arrays["disparity"], arrays["lr_valid"], camera)
            _check_row(cloud, statistics)
            filtered, filter_statistics = filter_cloud_support(arrays["disparity"], support)
            _check_row(cloud["filter"], filter_statistics)
            for name, selected in (("cloud", support), ("cloud_filtered", filtered)):
                saved = _load_arrays(directory / f"{name}.npz")
                y, x = np.nonzero(selected)
                expected_arrays = {"points": xyz[selected], "colors": left[:, :, ::-1][selected],
                    "pixels_xy": np.column_stack((x, y)), "depth": np.where(selected, xyz[:, :, 2], np.nan), "support": selected}
                for key, value in expected_arrays.items():
                    if not np.array_equal(value, saved[key], equal_nan=True):
                        raise ValueError(f"{name} arrays không khớp: {key}")
                with (directory / f"{name}.ply").open(encoding="ascii") as stream:
                    header = []
                    for line in stream:
                        header.append(line.strip())
                        if line.strip() == "end_header":
                            break
                    vertices = np.loadtxt(stream, ndmin=2)
                count = int(selected.sum())
                if f"element vertex {count}" not in header or not any("Assumed camera model" in line for line in header):
                    raise ValueError("PLY thiếu vertex count hoặc nhãn giả định.")
                if vertices.shape != (count, 6) or not np.allclose(vertices[:, :3], saved["points"], rtol=1e-7, atol=1e-7) or not np.array_equal(vertices[:, 3:], saved["colors"]):
                    raise ValueError("PLY không khớp tọa độ/màu đã lưu.")
                homogeneous = np.column_stack((saved["points"], np.ones(len(saved["points"]))))
                for side in ("left", "right"):
                    projected = homogeneous @ np.asarray(camera["P_" + side]).T
                    pixels = projected[:, :2] / projected[:, 2:]
                    expected_pixels = saved["pixels_xy"].astype(float).copy()
                    if side == "right":
                        expected_pixels[:, 0] -= arrays["disparity"][selected]
                    if not np.allclose(pixels, expected_pixels, atol=1e-4, rtol=0):
                        raise ValueError("Camera và disparity không tái chiếu nhất quán.")
        successful += 1
    comparisons = read_csv(run_dir / "common_domain_metrics.csv")
    recomputed_comparisons = common_domain_comparison(rows, run_dir / "cases", ground_truth, config["bad_pixel_threshold_px"])
    if len(comparisons) != len(recomputed_comparisons):
        raise ValueError("Bảng so sánh cùng domain thiếu/thừa cấu hình.")
    for recorded, expected in zip(comparisons, recomputed_comparisons):
        _check_row(recorded, expected)
    geometry = read_csv(run_dir / "geometry_metrics.csv")
    if [row["geometry_id"] for row in geometry] != [_geometry_id(value) for value in config["ransac_thresholds_px"]]:
        raise ValueError("Bảng geometry thiếu/thừa ngưỡng.")
    matches_path = run_dir / "geometry" / "matches.npz"
    matches = _load_arrays(matches_path) if matches_path.exists() else None
    if matches is not None:
        candidates = matches["knn"]
        accepted = candidates[:, 3] < config["ratio"] * candidates[:, 4]
        indices = candidates[accepted, :2].astype(np.int32)
        if not np.array_equal(accepted, matches["ratio_mask"]) or not np.array_equal(indices, matches["match_indices"]):
            raise ValueError("Ratio test không khớp KNN đã lưu.")
        for side, column in (("left", 0), ("right", 1)):
            points = matches["features_" + side][indices[:, column], :2].astype(np.float32)
            if not np.array_equal(points, matches["points_" + side]):
                raise ValueError("Tọa độ correspondence không khớp keypoint.")
    for row in geometry:
        if row["status"] == "ok":
            result = _load_arrays(run_dir / "geometry" / f"{row['geometry_id']}.npz")
            if matches is None or result["F"].shape != (3, 3) or not np.isfinite(result["F"]).all():
                raise ValueError("Thiếu matches hoặc F không hợp lệ.")
            singular = np.linalg.svd(result["F"], compute_uv=False)
            if singular[1] <= singular[0] * 1e-6 or singular[2] > singular[0] * 1e-6:
                raise ValueError("F đã lưu không có rank 2 ổn định.")
            if result["inliers"].dtype != bool or result["inliers"].shape != (len(matches["points_left"]),) or result["inliers"].sum() < 8:
                raise ValueError("Inlier mask không hợp lệ.")
            computed_errors = epipolar_errors(result["F"], matches["points_left"], matches["points_right"])
            for key, value in computed_errors.items():
                if not np.allclose(result[key], value, equal_nan=True):
                    raise ValueError("Sai số epipolar lưu không khớp.")
            _check_row(row, geometry_metrics(result, matches["points_left"], matches["points_right"]))
            _check_row(row, geometry_diagnostics(result["F"], ground_truth))
            _check_row(row, {"initial_matches": len(matches["knn"]), "detected_left": len(matches["features_left"]),
                            "detected_right": len(matches["features_right"]), "ransac_ms": float(result["ransac_ms"])})
        elif not row["error"]:
            raise ValueError("Geometry thất bại không có lý do.")
    cloud_count = sum(cloud["status"] == "ok" for cloud in manifest["point_clouds"].values())
    if manifest["status"] == "completed_with_assumed_camera" and (successful != len(cases) or any(row["status"] != "ok" for row in geometry) or cloud_count != 2):
        raise ValueError("Run khai báo hoàn tất nhưng thiếu đầu ra.")
    figures = _read_json(run_dir / "figures.json")
    if cloud_count:
        depths = np.concatenate([_load_arrays(run_dir / "cases" / name / "cloud_filtered.npz")["points"][:, 2]
                                 for name, cloud in manifest["point_clouds"].items() if cloud["status"] == "ok"])
        depth_min, depth_max = float(depths.min()), float(depths.max())
        if depth_min == depth_max:
            depth_max = depth_min + 1
        if figures.get("depth_display_limits") != [depth_min, depth_max]:
            raise ValueError("Thang depth không khớp khoảng chung của filtered predictions.")
    expected_figures = {"input_and_ground_truth", "bm_disparity_grid", "sgbm_disparity_grid"}
    if successful:
        expected_figures.add("metric_comparison")
    if matches is not None:
        expected_figures.add("initial_matches")
    if any(row["status"] == "ok" for row in geometry):
        expected_figures.add("geometry_validation")
    for row in geometry:
        if row["status"] == "ok":
            expected_figures.add(row["geometry_id"] + "_epipolar")
            if float(row["threshold_px"]) == 1.0:
                expected_figures.add("inlier_matches")
    for row in rows:
        if row["status"] == "ok":
            expected_figures.add(row["case_id"] + "_diagnostics")
    for case_id, cloud in manifest["point_clouds"].items():
        if cloud["status"] == "ok":
            expected_figures.update(case_id + suffix for suffix in ("_point_cloud", "_point_cloud_detail", "_depth", "_point_cloud_filtered", "_depth_filtered"))
    for name in expected_figures:
        relative = f"figures/{name}.png"
        if figures.get(name) != relative or relative not in actual_inventory or cv2.imread(str(run_dir / relative)) is None:
            raise ValueError(f"Thiếu/hỏng hình bắt buộc: {name}")
    validation = {"valid": True, "dense_cases": len(cases), "successful_dense_cases": successful,
        "geometry_cases": len(geometry), "point_clouds": cloud_count, "filtered_point_clouds": cloud_count,
        "geometry_warnings": sum(row.get("geometry_quality") == "warning" for row in geometry), "camera_model_kind": "assumed",
        "physical_3d_accuracy_established": False, "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(run_dir / "validation.json", validation)
    return validation


def _line_segment(line: np.ndarray, width: int, height: int):
    a, b, c = line
    intersections = []
    if abs(b) > 1e-12:
        for x in (0, width - 1):
            y = -(a * x + c) / b
            if 0 <= y <= height - 1:
                intersections.append((int(x), int(round(y))))
    if abs(a) > 1e-12:
        for y in (0, height - 1):
            x = -(b * y + c) / a
            if 0 <= x <= width - 1:
                intersections.append((int(round(x)), int(y)))
    intersections = list(dict.fromkeys(intersections))
    return intersections[:2] if len(intersections) >= 2 else None


def _match_image(left: np.ndarray, right: np.ndarray, points_left: np.ndarray, points_right: np.ndarray, seed: int) -> np.ndarray:
    canvas = np.concatenate((left, right), axis=1).copy()
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(points_left), min(80, len(points_left)), replace=False)
    for index in selected:
        color = tuple(int(value) for value in rng.integers(60, 256, 3))
        first = tuple(np.rint(points_left[index]).astype(int))
        second = tuple(np.rint(points_right[index] + [left.shape[1], 0]).astype(int))
        cv2.circle(canvas, first, 3, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, second, 3, color, 1, cv2.LINE_AA)
        cv2.line(canvas, first, second, color, 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def build_figures(run_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "run.json")
    left, right, ground_truth, _ = load_stereo_pair(run_dir / "inputs")
    config = manifest["config"]
    rows = read_csv(run_dir / "metrics.csv")
    geometry = read_csv(run_dir / "geometry_metrics.csv")
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    figures = {}

    def save(figure, name):
        figure.savefig(figure_dir / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
        figures[name] = f"figures/{name}.png"

    cmap = plt.get_cmap("turbo").with_extremes(bad="#dddddd")
    limits = (-10, 10)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for axis, image, title in zip(axes[:2], (left, right), ("Left / reference: frame10", "Right / target: frame11")):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    display = axes[2].imshow(np.ma.array(ground_truth["disparity"], mask=~ground_truth["valid"]), cmap=cmap, vmin=limits[0], vmax=limits[1])
    axes[2].set_title("GT disparity = -flow u (cropped frame)")
    axes[2].axis("off")
    figure.colorbar(display, ax=axes[2], label="Disparity (px)", extend="both", shrink=0.8)
    save(figure, "input_and_ground_truth")
    matches_path = run_dir / "geometry" / "matches.npz"
    if matches_path.exists():
        matches = _load_arrays(matches_path)
        candidates = matches["knn"][:, :2].astype(int)
        initial_left = matches["features_left"][candidates[:, 0], :2]
        initial_right = matches["features_right"][candidates[:, 1], :2]
        figure, axis = plt.subplots(figsize=(12, 5))
        axis.imshow(_match_image(left, right, initial_left, initial_right, config["seed"]))
        axis.set_title(f"KNN before Lowe test: {len(candidates)} matches (up to 80 shown)")
        axis.axis("off")
        save(figure, "initial_matches")
        for row in geometry:
            if row["status"] != "ok":
                continue
            result = _load_arrays(run_dir / "geometry" / f"{row['geometry_id']}.npz")
            inliers = result["inliers"]
            points_left, points_right = matches["points_left"][inliers], matches["points_right"][inliers]
            if float(row["threshold_px"]) == 1.0:
                figure, axis = plt.subplots(figsize=(12, 5))
                axis.imshow(_match_image(left, right, points_left, points_right, config["seed"]))
                axis.set_title(f"RANSAC inliers: {len(points_left)}/{len(matches['points_left'])} ratio-pass matches (up to 80 shown)")
                axis.axis("off")
                save(figure, "inlier_matches")
            views = [left.copy(), right.copy()]
            rng = np.random.default_rng(config["seed"])
            selected = rng.choice(len(points_left), min(20, len(points_left)), replace=False)
            for index in selected:
                first, second = points_left[index], points_right[index]
                lines = [result["F"].T @ [*second, 1], result["F"] @ [*first, 1]]
                color = tuple(int(value) for value in rng.integers(50, 256, 3))
                for view, line, point in zip(views, lines, (first, second)):
                    segment = _line_segment(line, view.shape[1], view.shape[0])
                    if segment:
                        cv2.line(view, *segment, color, 1, cv2.LINE_AA)
                    cv2.circle(view, tuple(np.rint(point).astype(int)), 4, color, 2, cv2.LINE_AA)
            figure, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
            for axis, view, side in zip(axes, views, ("Left", "Right")):
                axis.imshow(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
                axis.set_title(f"{side}: RANSAC threshold {row['threshold_px']} px")
                axis.axis("off")
            figure.suptitle(f"USAC_MAGSAC | quality: {row['geometry_quality']} | independent GT epipolar mean: {float(row['gt_domain_mean_epipolar_px']):.3f} px")
            save(figure, f"{row['geometry_id']}_epipolar")

    successful_geometry = [row for row in geometry if row["status"] == "ok"]
    if successful_geometry:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        labels = [row["geometry_id"] for row in successful_geometry]
        positions = np.arange(len(labels))
        for offset, field, label in ((-0.18, "inlier_mean_epipolar_px", "SIFT inliers"),
                                     (0.18, "gt_domain_mean_epipolar_px", "Independent GT domain")):
            axes[0].bar(positions + offset, [float(row[field]) for row in successful_geometry], width=0.36, label=label)
        axes[0].set(ylabel="Mean symmetric epipolar error (px)", xticks=positions, xticklabels=labels)
        axes[0].legend()
        for side in ("left", "right"):
            axes[1].plot(labels, [float(row[f"{side}_angle_p95_deg"]) for row in successful_geometry], "o-", label=side)
        axes[1].axhline(2, color="red", linestyle="--", label="Project diagnostic limit: 2°")
        axes[1].set(ylabel="p95 absolute epipolar angle (degrees)")
        axes[1].legend()
        figure.suptitle("Estimated F diagnostics; rectified GT reference: horizontal lines, zero residual")
        save(figure, "geometry_validation")

    for method in config["methods"]:
        figure, axes = plt.subplots(len(config["num_disparities"]), len(config["block_sizes"]), squeeze=False,
                                    figsize=(4.5 * len(config["block_sizes"]), 4 * len(config["num_disparities"])), layout="constrained")
        for index, number in enumerate(config["num_disparities"]):
            for column, block in enumerate(config["block_sizes"]):
                axis = axes[index, column]
                row = next(item for item in rows if item["case_id"] == f"{method}_n{number}_b{block}")
                axis.set_title(f"{method.upper()} | ndisp={number}, block={block}")
                axis.axis("off")
                if row["status"] != "ok":
                    axis.text(0.05, 0.5, "FAILED: " + row["error"], wrap=True)
                    continue
                arrays = _load_arrays(run_dir / "cases" / row["case_id"] / "disparity.npz")
                display = axis.imshow(np.ma.array(arrays["disparity"], mask=~arrays["lr_valid"]), cmap=cmap, vmin=limits[0], vmax=limits[1])
                figure.colorbar(display, ax=axis, label="Disparity (px)", extend="both", shrink=0.7)
        figure.suptitle("LR-consistent disparity; gray = invalid; display saturates outside [-10, 10] px")
        save(figure, f"{method}_disparity_grid")
    for row in rows:
        if row["status"] != "ok":
            continue
        arrays = _load_arrays(run_dir / "cases" / row["case_id"] / "disparity.npz")
        evaluation = ground_truth["evaluation_domain"] & arrays["lr_valid"]
        errors = np.abs(arrays["disparity"] - ground_truth["disparity"])
        figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for axis, name, title in zip(axes[:2], ("raw_valid", "lr_valid"), ("Raw valid mask", "LR-consistent mask (not GT occlusion)")):
            axis.imshow(arrays[name], cmap="gray", vmin=0, vmax=1)
            axis.set_title(title)
            axis.axis("off")
        display = axes[2].imshow(np.ma.array(errors, mask=~evaluation), vmin=0, vmax=3, cmap="magma")
        axes[2].set_title("Absolute disparity error")
        axes[2].axis("off")
        figure.colorbar(display, ax=axes[2], label="Error (px)", extend="max", shrink=0.8)
        figure.suptitle(row["case_id"])
        save(figure, f"{row['case_id']}_diagnostics")
    successful = [row for row in rows if row["status"] == "ok"]
    if successful:
        figure, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
        labels = [row["case_id"] for row in successful]
        fields = ("stereo_pipeline_ms", "consistent_valid_ratio_pct", "rmse_px", "coverage_pct")
        titles = ("Dense pipeline median (ms); whiskers = min/max", "LR valid (% of full image)",
                  "GT RMSE on valid intersection (px)", "Coverage (% of fixed GT domain)")
        for axis, field, title in zip(axes.ravel(), fields, titles):
            values = [float(row[field]) if row[field] else np.nan for row in successful]
            axis.bar(labels, values)
            if field == "stereo_pipeline_ms":
                lower = np.array(values) - [float(row["stereo_pipeline_min_ms"]) for row in successful]
                upper = np.array([float(row["stereo_pipeline_max_ms"]) for row in successful]) - values
                axis.errorbar(labels, values, yerr=[lower, upper], fmt="none", ecolor="black", capsize=3)
            axis.set_title(title)
            axis.tick_params(axis="x", rotation=90)
        save(figure, "metric_comparison")
    clouds = {name: _load_arrays(run_dir / "cases" / name / "cloud.npz")
              for name, cloud in manifest["point_clouds"].items() if cloud["status"] == "ok"}
    if clouds:
        filtered_clouds = {name: _load_arrays(run_dir / "cases" / name / "cloud_filtered.npz") for name in clouds}
        combined = np.concatenate([cloud["points"] for cloud in clouds.values()])
        minimum, maximum = combined.min(axis=0), combined.max(axis=0)
        center = (minimum + maximum) / 2
        radius = max(float((maximum - minimum).max()) / 2, 1e-6) * 1.03
        filtered_points = np.concatenate([cloud["points"] for cloud in filtered_clouds.values()])
        detail_center = (filtered_points.min(axis=0) + filtered_points.max(axis=0)) / 2
        detail_radius = max(float(np.ptp(filtered_points, axis=0).max()) / 2, 1e-6) * 1.03
        depth_min, depth_max = float(filtered_points[:, 2].min()), float(filtered_points[:, 2].max())
        if depth_max == depth_min:
            depth_max = depth_min + 1
        figures["depth_display_limits"] = [depth_min, depth_max]
        figures["cloud_detail_bounds_source"] = "Union of filtered estimated clouds; no GT used for filtering or display bounds."
        for name, cloud in clouds.items():
            for variant in ("full", "detail", "filtered"):
                displayed = filtered_clouds[name] if variant == "filtered" else cloud
                view_center, view_radius = (center, radius) if variant == "full" else (detail_center, detail_radius)
                within_view = np.all(np.abs(displayed["points"] - view_center) <= view_radius, axis=1)
                candidates = np.flatnonzero(within_view) if variant == "detail" else np.arange(len(displayed["points"]))
                step = max(1, math.ceil(len(candidates) / 20000))
                selected = candidates[::step]
                if variant != "detail":
                    extrema = np.r_[np.argmin(displayed["points"], axis=0), np.argmax(displayed["points"], axis=0)]
                    selected = np.unique(np.r_[selected, extrema])
                preview = displayed["points"][selected]
                colors = displayed["colors"][selected] / 255.0
                figure = plt.figure(figsize=(17, 6), layout="constrained")
                for index, (elevation, azimuth) in enumerate(((15, -90), (25, -45), (70, -90))):
                    axis = figure.add_subplot(1, 3, index + 1, projection="3d")
                    axis.scatter(*preview.T, c=colors, s=1, depthshade=False)
                    axis.set(xlim=(view_center[0] - view_radius, view_center[0] + view_radius),
                             ylim=(view_center[1] - view_radius, view_center[1] + view_radius),
                             zlim=(view_center[2] - view_radius, view_center[2] + view_radius),
                             xlabel="X", ylabel="Y", zlabel="Z")
                    axis.tick_params(labelsize=8)
                    axis.locator_params(nbins=4)
                    axis.set_box_aspect((1, 1, 1))
                    axis.view_init(elev=elevation, azim=azimuth)
                view_label = ("Raw full extent; all extrema included" if variant == "full" else
                              f"{variant.title()}: shared filtered-prediction bounds; {int((~within_view).sum()):,} vertices outside view")
                figure.suptitle(f"{name}: Assumed camera model; normalized units\n{view_label}; {len(preview):,}/{len(displayed['points']):,} vertices shown; raw PLY preserved")
                suffix = {"full": "_point_cloud", "detail": "_point_cloud_detail", "filtered": "_point_cloud_filtered"}[variant]
                save(figure, name + suffix)
            for suffix, displayed in (("_depth", cloud), ("_depth_filtered", filtered_clouds[name])):
                figure, axis = plt.subplots(figsize=(7, 5), layout="constrained")
                display = axis.imshow(np.ma.masked_invalid(displayed["depth"]), cmap="viridis", vmin=depth_min, vmax=depth_max)
                outside = int(((displayed["points"][:, 2] < depth_min) | (displayed["points"][:, 2] > depth_max)).sum())
                axis.set_title(f"{name}: {'filtered' if suffix.endswith('filtered') else 'raw'} depth\nShared scale; {outside} values saturated")
                axis.axis("off")
                figure.colorbar(display, ax=axis, label="Depth (normalized baseline units)", extend="both")
                save(figure, name + suffix)
    figures["display_convention"] = "Disparity display fixed to [-10,10] px; values outside are saturated, raw arrays unchanged. Error display [0,3] px."
    _write_json(run_dir / "figures.json", figures)
    manifest["artifacts"] = _inventory(run_dir)
    _write_json(run_dir / "run.json", manifest)
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description="Part B: Venus stereo matching and pinhole reconstruction.")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        config = SMOKE_CONFIG if args.profile == "smoke" else DEFAULT_CONFIG
        run_dir = run_experiments(args.input_dir, args.output, config)
        status = _read_json(run_dir / "run.json")["status"]
        print(json.dumps({"run_dir": str(run_dir), "status": status, "camera_model_kind": "assumed", "physical_3d_accuracy_established": False}))
        return 0 if status == "completed_with_assumed_camera" else 1
    except (OSError, ValueError, cv2.error) as error:
        print(f"Part B failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
