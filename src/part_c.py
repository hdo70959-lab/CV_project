"""Ghép bốn ảnh bằng SIFT/ORB/AKAZE và lưu bằng chứng cho từng cấu hình."""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

DEFAULT_INPUT = Path("data/part_c")
DEFAULT_OUTPUT = Path("results/part_c")
ANCHOR_INDEX = 2
DEFAULT_CONFIG = {
    "profile": "full",
    "methods": ["sift", "orb", "akaze"],
    "ratios": [0.70, 0.75, 0.80],
    "thresholds_px": [2.0, 3.0, 5.0],
    "max_features": 5000,
    "max_canvas_pixels": 40_000_000,
    "confidence": 0.995,
    "max_iters": 5000,
}
SMOKE_CONFIG = dict(DEFAULT_CONFIG, profile="smoke", ratios=[0.75], thresholds_px=[3.0])
TIMING_SCOPE = (
    "grayscale_and_detector_creation_through_cropped_panorama_excluding_io_metrics_figures"
)
METRIC_FIELDS = [
    "case_id",
    "method",
    "ratio",
    "threshold_px",
    "status",
    "error",
    "stage",
    "detected_keypoints",
    "used_keypoints",
    "initial_matches",
    "ratio_pass_matches",
    "inliers",
    "inlier_ratio_pct",
    "rmse_px",
    "squared_error_sum",
    "canvas_width",
    "canvas_height",
    "total_stitching_ms",
    "elapsed_to_failure_ms",
    "timing_kind",
]
PAIR_FIELDS = [
    "case_id",
    "source_image",
    "destination_image",
    "status",
    "error",
    "initial_matches",
    "ratio_pass_matches",
    "inliers",
    "inlier_ratio_pct",
    "rmse_px",
    "squared_error_sum",
    "source_hull_fraction",
    "destination_hull_fraction",
]


def _digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Không ghi được ảnh: {path}")


def load_sequence(input_dir: Path) -> tuple[list[np.ndarray], dict]:
    """Giữ thứ tự metadata và độ phân giải gốc; từ chối nguồn bị thay đổi."""
    input_dir = Path(input_dir)
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    records = metadata.get("images", [])
    if len(records) != 4:
        raise ValueError("Part C cần đúng bốn ảnh trong metadata.")
    names = [record["filename"] for record in records]
    if len(set(names)) != 4 or any(Path(name).name != name or not name for name in names):
        raise ValueError("Tên ảnh phải duy nhất và nằm trực tiếp trong thư mục input.")
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".heif", ".heic"}
    present = {path.name for path in input_dir.iterdir() if path.suffix.lower() in image_extensions}
    if present != set(names):
        raise ValueError("Ảnh trong thư mục không khớp bốn ảnh được khai báo.")
    images = []
    for record in records:
        path = input_dir / record["filename"]
        if _digest(path) != record["sha256"]:
            raise ValueError(f"SHA-256 không khớp: {path.name}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {path.name}")
        if list(image.shape[1::-1]) != record["size"]:
            raise ValueError(f"Kích thước ảnh không khớp metadata: {path.name}")
        images.append(image)
    return images, metadata


def _measurement_syntax(source: str) -> str:
    """So sánh logic phép đo, bỏ chú thích và các hàm tìm run/dựng hình."""
    tree = ast.parse(source)
    excluded = {
        "_measurement_syntax",
        "_matches_recorded_source",
        "find_matching_run",
        "build_figures",
        "_build_blending_figures",
    }
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name in excluded)
        and not (isinstance(node, ast.Import) and all(alias.name == "ast" for alias in node.names))
    ]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def _matches_recorded_source(run_dir: Path, manifest: dict, source: str) -> bool:
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    snapshots = (
        ("source_code.py", "source_sha256"),
        ("finalization_source.py", "finalization_source_sha256"),
    )
    for filename, field in snapshots:
        expected = manifest.get(field)
        if expected == source_hash:
            return True
        path = run_dir / filename
        if expected is not None and path.is_file() and _digest(path) == expected:
            if _measurement_syntax(path.read_text(encoding="utf-8")) == _measurement_syntax(source):
                return True
    return False


def find_matching_run(input_dir: Path, output_root: Path, config: dict) -> Path:
    """Chỉ tự chọn run khớp dữ liệu, code producer/finalization và dependency."""
    load_sequence(input_dir)
    metadata_hash = _digest(Path(input_dir) / "metadata.json")
    current_source = Path(__file__).read_text(encoding="utf-8")
    packages = {
        name: version(name) for name in ("numpy", "opencv-contrib-python-headless", "matplotlib")
    }
    candidates = []
    for path in sorted(Path(output_root).glob("*/run.json"), reverse=True):
        manifest = json.loads(path.read_text())
        if (
            manifest.get("config") == config
            and manifest.get("status") in ("complete", "complete_with_failures")
            and manifest.get("input_metadata_sha256") == metadata_hash
            and _matches_recorded_source(path.parent, manifest, current_source)
            and manifest.get("environment", {}).get("packages") == packages
            and manifest.get("environment", {}).get("opencv") == cv2.__version__
        ):
            candidates.append(path.parent)
    if not candidates:
        raise FileNotFoundError(
            "Chưa có run khớp dữ liệu/cấu hình/code/môi trường hiện tại; hãy chạy thí nghiệm trước."
        )
    reviewed = [
        path
        for path in candidates
        if json.loads((path / "selection.json").read_text())["status"] == "reviewed"
    ]
    return (reviewed or candidates)[0]


def create_detector(method: str, max_features: int):
    if method == "sift":
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    if method == "orb":
        return cv2.ORB_create(nfeatures=max_features, WTA_K=2), cv2.NORM_HAMMING
    if method == "akaze":
        return (
            cv2.xfeatures2d.AKAZE_create(threshold=0.001, max_points=max_features),
            cv2.NORM_HAMMING,
        )
    raise ValueError(f"Detector không được hỗ trợ: {method}")


def limit_features(keypoints, descriptors, max_features: int) -> dict:
    """Cắt cả descriptor theo cùng chỉ số, kể cả khi SIFT trả thừa điểm đồng hạng."""
    order = sorted(range(len(keypoints)), key=lambda index: -keypoints[index].response)[
        :max_features
    ]
    return {
        "keypoints": [keypoints[index] for index in order],
        "descriptors": descriptors[order] if descriptors is not None else None,
        "detected_count": len(keypoints),
    }


def match_features(source: dict, destination: dict, norm: int, ratio: float) -> dict:
    if not 0 < ratio < 1:
        raise ValueError("Lowe ratio phải nằm trong (0, 1).")
    result = {
        "initial_matches": 0,
        "knn": np.empty((0, 5), np.float64),
        "match_indices": np.empty((0, 2), np.int32),
        "source_points": np.empty((0, 2), np.float32),
        "destination_points": np.empty((0, 2), np.float32),
    }
    if source["descriptors"] is None or destination["descriptors"] is None:
        return result
    if len(source["descriptors"]) == 0 or len(destination["descriptors"]) < 2:
        return result
    neighbours = cv2.BFMatcher(norm, crossCheck=False).knnMatch(
        source["descriptors"], destination["descriptors"], k=2
    )
    complete = [pair for pair in neighbours if len(pair) == 2]
    result["initial_matches"] = len(complete)
    result["knn"] = np.asarray(
        [[a.queryIdx, a.trainIdx, b.trainIdx, a.distance, b.distance] for a, b in complete],
        dtype=np.float64,
    ).reshape(-1, 5)
    matches = [a for a, b in complete if a.distance < ratio * b.distance]
    result["match_indices"] = np.asarray(
        [[match.queryIdx, match.trainIdx] for match in matches], np.int32
    ).reshape(-1, 2)
    result["source_points"] = np.asarray(
        [source["keypoints"][m.queryIdx].pt for m in matches], np.float32
    ).reshape(-1, 2)
    result["destination_points"] = np.asarray(
        [destination["keypoints"][m.trainIdx].pt for m in matches], np.float32
    ).reshape(-1, 2)
    return result


def estimate_homography(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    threshold: float,
    permutation_seed: int | None = None,
    confidence: float = 0.995,
    max_iters: int = 5000,
) -> dict:
    source = np.asarray(source_points, np.float32).reshape(-1, 2)
    destination = np.asarray(destination_points, np.float32).reshape(-1, 2)
    if (
        len(source) != len(destination)
        or len(source) < 4
        or not np.isfinite(source).all()
        or not np.isfinite(destination).all()
    ):
        raise ValueError("Homography cần ít nhất bốn correspondence hữu hạn.")
    for points in (source, destination):
        if np.linalg.matrix_rank(points.astype(np.float64) - points.mean(axis=0)) < 2:
            raise ValueError("Correspondence đồng tuyến hoặc trùng nhau.")
    permutation = np.arange(len(source))
    if permutation_seed is not None:
        permutation = np.random.default_rng(permutation_seed).permutation(len(source))
    H, shuffled_mask = cv2.findHomography(
        source[permutation],
        destination[permutation],
        cv2.RANSAC,
        threshold,
        maxIters=max_iters,
        confidence=confidence,
    )
    if (
        H is None
        or shuffled_mask is None
        or not np.isfinite(H).all()
        or np.linalg.matrix_rank(H) < 3
    ):
        raise ValueError("RANSAC không tìm được homography hợp lệ.")
    mask = np.zeros(len(source), dtype=bool)
    mask[permutation] = shuffled_mask.ravel().astype(bool)
    if mask.sum() < 4:
        raise ValueError("Homography có ít hơn bốn inlier.")
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return {"H": H, "inlier_mask": mask, "permutation": permutation}


def _corners(shape) -> np.ndarray:
    height, width = shape[:2]
    return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], np.float64)


def canvas_geometry(shapes: list, transforms: list[np.ndarray], max_canvas_pixels: int) -> dict:
    if not shapes or len(shapes) != len(transforms):
        raise ValueError("Số ảnh và số biến đổi phải khớp.")
    projected = []
    for shape, matrix in zip(shapes, transforms):
        H = np.asarray(matrix, np.float64)
        if H.shape != (3, 3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
            raise ValueError("Homography không hữu hạn hoặc suy biến.")
        corners = _corners(shape)
        homogeneous = np.column_stack([corners, np.ones(4)]) @ H.T
        denominator = homogeneous[:, 2]
        if not (np.all(denominator > 1e-10) or np.all(denominator < -1e-10)):
            raise ValueError("Projective horizon đi qua miền ảnh.")
        points = homogeneous[:, :2] / denominator[:, None]
        if not np.isfinite(points).all():
            raise ValueError("Góc ảnh sau chiếu không hữu hạn.")
        projected.append(points)
    points = np.concatenate(projected)
    lower = np.floor(points.min(axis=0))
    upper = np.ceil(points.max(axis=0))
    width, height = map(int, upper - lower)
    if width < 1 or height < 1 or width * height > max_canvas_pixels:
        raise ValueError(
            f"Canvas {width}×{height} vượt giới hạn {max_canvas_pixels} pixel hoặc rỗng."
        )
    translation = np.array([[1.0, 0.0, -lower[0]], [0.0, 1.0, -lower[1]], [0.0, 0.0, 1.0]])
    return {
        "canvas_size": [width, height],
        "translation": translation.tolist(),
        "projected_corners": [points.tolist() for points in projected],
    }


def blend_images(images: list[np.ndarray], transforms: list[np.ndarray], max_canvas_pixels: int):
    geometry = canvas_geometry([image.shape for image in images], transforms, max_canvas_pixels)
    width, height = geometry["canvas_size"]
    translation = np.asarray(geometry["translation"])
    numerator = np.zeros((height, width, 3), np.float32)
    denominator = np.zeros((height, width), np.float32)
    for image, H in zip(images, transforms):
        source_mask = np.pad(np.ones(image.shape[:2], np.uint8), 1)
        weights = cv2.distanceTransform(source_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        weights /= weights.max()
        matrix = translation @ H
        # Warp màu đã nhân alpha để nội suy với nền zero không làm tối pixel biên.
        warped_color = cv2.warpPerspective(
            image.astype(np.float32) * weights[:, :, None],
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        numerator += warped_color
        del warped_color
        denominator += cv2.warpPerspective(
            weights, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
    mask = denominator > 1e-8
    if not mask.any():
        raise ValueError("Panorama không có vùng hợp lệ.")
    np.divide(numerator, denominator[:, :, None], out=numerator, where=mask[:, :, None])
    panorama = np.rint(np.clip(numerator, 0, 255)).astype(np.uint8)
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    x0, y0, x1, y1 = int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1
    geometry["crop_offset"] = [x0, y0]
    geometry["output_size"] = [x1 - x0, y1 - y0]
    return panorama[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy(), geometry


def blend_multiband(
    images: list[np.ndarray],
    transforms: list[np.ndarray],
    max_canvas_pixels: int,
    num_bands: int = 5,
):
    """Laplacian pyramid của OpenCV; chỉ thay blending, giữ nguyên hệ tọa độ và H.

    Màu ngoài miền ảnh được phản chiếu để pyramid không hút nền zero vào biên.
    Mask hỗ trợ hình học vẫn giới hạn chính xác nội dung được xuất.
    """
    if type(num_bands) is not int or not 1 <= num_bands <= 8:
        raise ValueError("Số band phải là số nguyên từ 1 đến 8.")
    geometry = canvas_geometry([image.shape for image in images], transforms, max_canvas_pixels)
    width, height = geometry["canvas_size"]
    translation = np.asarray(geometry["translation"])
    blender = cv2.detail_MultiBandBlender(0, num_bands, cv2.CV_32F)
    blender.prepare((0, 0, width, height))
    union = np.zeros((height, width), bool)
    for image, H in zip(images, transforms):
        matrix = translation @ H
        support = (
            cv2.warpPerspective(
                np.ones(image.shape[:2], np.float32),
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            > 1e-8
        )
        union |= support
        color = cv2.warpPerspective(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        # Tăng độ chính xác dưới pixel trong pyramid int16, vẫn còn dư miền giá trị.
        blender.feed(color.astype(np.int16) * 32, support.astype(np.uint8) * 255, (0, 0))
    blended, _ = blender.blend(None, None)
    panorama = np.rint(np.clip(blended.astype(np.float32) / 32, 0, 255)).astype(np.uint8)
    panorama[~union] = 0
    rows = np.flatnonzero(union.any(axis=1))
    columns = np.flatnonzero(union.any(axis=0))
    if not len(rows) or not len(columns):
        raise ValueError("Panorama không có vùng hợp lệ.")
    x0, y0, x1, y1 = int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1
    geometry.update(crop_offset=[x0, y0], output_size=[x1 - x0, y1 - y0])
    return panorama[y0:y1, x0:x1].copy(), union[y0:y1, x0:x1].copy(), geometry


def pair_metrics(source, destination, H, inlier_mask, source_shape, destination_shape) -> dict:
    mask = np.asarray(inlier_mask, bool)
    count = int(mask.sum())
    if count == 0:
        raise ValueError("Không có inlier để đo sai số.")
    source = np.asarray(source, np.float32).reshape(-1, 2)
    destination = np.asarray(destination, np.float32).reshape(-1, 2)
    predicted = cv2.perspectiveTransform(
        source[mask].astype(np.float64).reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    squared_error_sum = float(np.square(predicted - destination[mask]).sum())
    if not math.isfinite(squared_error_sum):
        raise ValueError("Sai số chiếu không hữu hạn.")
    fractions = []
    for points, shape in ((source[mask], source_shape), (destination[mask], destination_shape)):
        fractions.append(float(cv2.contourArea(cv2.convexHull(points))) / (shape[0] * shape[1]))
    return {
        "ratio_pass_matches": len(source),
        "inliers": count,
        "inlier_ratio_pct": 100.0 * count / len(source),
        "rmse_px": math.sqrt(squared_error_sum / count),
        "squared_error_sum": squared_error_sum,
        "source_hull_fraction": fractions[0],
        "destination_hull_fraction": fractions[1],
    }


def summarize_pairs(rows: list[dict]) -> dict:
    result = {
        name: sum(row[name] for row in rows)
        for name in ("initial_matches", "ratio_pass_matches", "inliers", "squared_error_sum")
    }
    result["inlier_ratio_pct"] = 100.0 * result["inliers"] / result["ratio_pass_matches"]
    result["rmse_px"] = math.sqrt(result["squared_error_sum"] / result["inliers"])
    return result


def run_pipeline(
    images: list[np.ndarray], method: str, ratio: float, threshold: float, config: dict
) -> dict:
    result = {
        "status": "ok",
        "error": "",
        "stage": "extraction",
        "features": [],
        "pairs": [],
        "transforms": [np.eye(3) for _ in images],
    }
    start = time.perf_counter_ns()
    try:
        detector, norm = create_detector(method, config["max_features"])
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            result["features"].append(
                limit_features(keypoints, descriptors, config["max_features"])
            )
        for index in range(4):
            if index == ANCHOR_INDEX:
                continue
            result["stage"] = f"matching_{index + 1}_to_3"
            pair = match_features(
                result["features"][index], result["features"][ANCHOR_INDEX], norm, ratio
            )
            pair.update(source_index=index, destination_index=ANCHOR_INDEX)
            result["pairs"].append(pair)
            result["stage"] = f"homography_{index + 1}_to_3"
            pair.update(
                estimate_homography(
                    pair["source_points"],
                    pair["destination_points"],
                    threshold,
                    confidence=config["confidence"],
                    max_iters=config["max_iters"],
                )
            )
            result["transforms"][index] = pair["H"]
        result["stage"] = "warping_and_blending"
        panorama, mask, geometry = blend_images(
            images, result["transforms"], config["max_canvas_pixels"]
        )
        result.update(panorama=panorama, mask=mask, geometry=geometry, stage="complete")
        result["total_stitching_ms"] = (time.perf_counter_ns() - start) / 1e6
    except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
        result.update(
            status="error",
            error=f"{type(error).__name__}: {error}",
            elapsed_to_failure_ms=(time.perf_counter_ns() - start) / 1e6,
        )
    return result


def _case_id(method: str, ratio: float, threshold: float) -> str:
    return f"{method}_r{ratio:.2f}_t{threshold:g}".replace(".", "p")


def _cases(config: dict) -> list[dict]:
    return [
        {
            "case_id": _case_id(method, ratio, threshold),
            "method": method,
            "ratio": ratio,
            "threshold_px": threshold,
        }
        for method in config["methods"]
        for ratio in config["ratios"]
        for threshold in config["thresholds_px"]
    ]


def _validate_config(config: dict) -> None:
    if not config["methods"] or len(set(config["methods"])) != len(config["methods"]):
        raise ValueError("Danh sách phương pháp rỗng hoặc bị lặp.")
    for method in config["methods"]:
        create_detector(method, config["max_features"])
    for name in ("max_features", "max_canvas_pixels", "max_iters"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} phải là số nguyên dương.")
    if not 0 < config["confidence"] < 1:
        raise ValueError("Confidence phải nằm trong (0, 1).")
    for name, predicate in (("ratios", lambda x: 0 < x < 1), ("thresholds_px", lambda x: x > 0)):
        values = config[name]
        if (
            not values
            or len(set(values)) != len(values)
            or any(not math.isfinite(x) or not predicate(x) for x in values)
        ):
            raise ValueError(f"{name} không hợp lệ.")
    ids = [case["case_id"] for case in _cases(config)]
    if len(ids) != len(set(ids)):
        raise ValueError("Tham số tạo ID cấu hình trùng nhau; cần tăng độ phân giải tên ID.")


def _save_case(
    run_dir: Path, case: dict, result: dict, images: list[np.ndarray]
) -> tuple[dict, list[dict]]:
    directory = run_dir / "cases" / case["case_id"]
    directory.mkdir()
    row = dict.fromkeys(METRIC_FIELDS)
    row.update(
        case,
        status=result["status"],
        error=result["error"],
        stage=result["stage"],
        timing_kind="single_sample",
    )
    for key in ("total_stitching_ms", "elapsed_to_failure_ms"):
        row[key] = result.get(key)
    row["detected_keypoints"] = sum(feature["detected_count"] for feature in result["features"])
    row["used_keypoints"] = sum(len(feature["keypoints"]) for feature in result["features"])
    for index, feature in enumerate(result["features"]):
        path = run_dir / "features" / f"{case['method']}_{index + 1}.npz"
        if not path.exists():
            points = np.asarray([point.pt for point in feature["keypoints"]], np.float32).reshape(
                -1, 2
            )
            descriptions = feature["descriptors"]
            np.savez_compressed(
                path,
                points=points,
                descriptors=(
                    descriptions if descriptions is not None else np.empty((0, 0), np.uint8)
                ),
                responses=np.asarray([point.response for point in feature["keypoints"]]),
                detected_count=feature["detected_count"],
            )
    pair_rows = []
    for pair in result["pairs"]:
        pair_row = dict.fromkeys(PAIR_FIELDS)
        pair_row.update(
            case_id=case["case_id"],
            source_image=pair["source_index"] + 1,
            destination_image=3,
            status="ok" if "H" in pair else "error",
            error="" if "H" in pair else result["error"],
            initial_matches=pair["initial_matches"],
            ratio_pass_matches=len(pair["source_points"]),
        )
        if "H" in pair:
            pair_row.update(
                pair_metrics(
                    pair["source_points"],
                    pair["destination_points"],
                    pair["H"],
                    pair["inlier_mask"],
                    images[pair["source_index"]].shape,
                    images[ANCHOR_INDEX].shape,
                )
            )
        arrays = {
            key: pair[key]
            for key in (
                "knn",
                "match_indices",
                "source_points",
                "destination_points",
                "H",
                "inlier_mask",
                "permutation",
            )
            if key in pair
        }
        np.savez_compressed(directory / f"pair_{pair['source_index'] + 1}_to_3.npz", **arrays)
        pair_rows.append(pair_row)
    if result["status"] == "ok":
        row.update(summarize_pairs(pair_rows))
        row["canvas_width"], row["canvas_height"] = result["geometry"]["canvas_size"]
        _save_image(directory / "panorama.png", result["panorama"])
        _save_image(directory / "mask.png", result["mask"].astype(np.uint8) * 255)
        _write_json(
            directory / "geometry.json",
            dict(result["geometry"], transforms=[H.tolist() for H in result["transforms"]]),
        )
    _write_json(directory / "case.json", row)
    return row, pair_rows


def _inventory(run_dir: Path) -> dict:
    return {
        str(path.relative_to(run_dir)): _digest(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.name not in {"run.json", "validation.json"}
        and path.suffix != ".tmp"
    }


def _refresh_validation(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run.json").read_text())
    manifest["artifacts"] = _inventory(run_dir)
    _write_json(run_dir / "run.json", manifest)
    validation = validate_run(run_dir)
    _write_json(run_dir / "validation.json", validation)
    if not validation["valid"]:
        raise RuntimeError("Artifact không hợp lệ: " + "; ".join(validation["errors"][:5]))
    return validation


def run_experiments(input_dir: Path, output_root: Path, config: dict) -> Path:
    config = copy.deepcopy(config)
    _validate_config(config)
    images, metadata = load_sequence(input_dir)
    packages = {
        name: version(name) for name in ("numpy", "opencv-contrib-python-headless", "matplotlib")
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True)
    for name in ("inputs", "features", "cases", "figures"):
        (run_dir / name).mkdir()
    for record in metadata["images"]:
        shutil.copyfile(
            Path(input_dir) / record["filename"], run_dir / "inputs" / record["filename"]
        )
    shutil.copyfile(Path(input_dir) / "metadata.json", run_dir / "inputs/metadata.json")
    shutil.copyfile(Path(__file__), run_dir / "source_code.py")
    manifest = {
        "run_id": run_id,
        "status": "running",
        "config": config,
        "input_metadata": metadata,
        "input_metadata_sha256": _digest(Path(input_dir) / "metadata.json"),
        "anchor_image": 3,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": packages,
            "opencv": cv2.__version__,
            "opencv_threads": 1,
            "opencl": False,
        },
        "color_space": "BGR arrays; PNG encoded by OpenCV; RGB for Matplotlib",
        "timing_scope": TIMING_SCOPE,
        "source_sha256": _digest(Path(__file__)),
        "expected_case_count": len(_cases(config)),
        "selection_status": "pending_visual_review",
    }
    lockfile = Path(__file__).resolve().parents[1] / "uv.lock"
    if lockfile.is_file():
        manifest["uv_lock_sha256"] = _digest(lockfile)
    _write_json(run_dir / "run.json", manifest)
    previous_threads, previous_opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    rows, pairs = [], []
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        for case in _cases(config):
            print(f"[{run_id}] {case['case_id']}", flush=True)
            result = run_pipeline(
                images, case["method"], case["ratio"], case["threshold_px"], config
            )
            row, pair_rows = _save_case(run_dir, case, result, images)
            rows.append(row)
            pairs.extend(pair_rows)
            _write_csv(run_dir / "metrics.csv", rows, METRIC_FIELDS)
            _write_csv(run_dir / "pairs.csv", pairs, PAIR_FIELDS)
            del result
        _write_json(
            run_dir / "selection.json",
            {"run_id": run_id, "status": "pending_visual_review", "choices": {}},
        )
        outcome = (
            "complete" if all(row["status"] == "ok" for row in rows) else "complete_with_failures"
        )
        manifest["status"] = "building_figures"
        manifest["successful_case_count"] = sum(row["status"] == "ok" for row in rows)
        _write_json(run_dir / "run.json", manifest)
        build_figures(run_dir)
        _refresh_validation(run_dir)
        manifest = json.loads((run_dir / "run.json").read_text())
        manifest["status"] = outcome
        _write_json(run_dir / "run.json", manifest)
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        _write_json(run_dir / "run.json", manifest)
        raise
    finally:
        cv2.setNumThreads(previous_threads)
        cv2.ocl.setUseOpenCL(previous_opencl)
    return run_dir


def _robustness(
    run_dir: Path,
    choices: dict,
    images: list[np.ndarray],
    config: dict,
    directory: Path | None = None,
) -> list[dict]:
    directory = run_dir / "robustness" if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, choice in choices.items():
        case_dir = run_dir / "cases" / choice["case_id"]
        case = json.loads((case_dir / "case.json").read_text())
        for seed in (42, 123, 2026):
            matrices = [np.eye(3) for _ in images]
            seed_rows, arrays = [], {}
            for source_index in (0, 1, 3):
                row = {
                    "case_id": case["case_id"],
                    "method": method,
                    "permutation_seed": seed,
                    "source_image": source_index + 1,
                    "destination_image": 3,
                    "status": "ok",
                    "error": "",
                    "inliers": None,
                    "inlier_ratio_pct": None,
                    "rmse_px": None,
                    "corner_rmse_px": None,
                    "corner_max_deviation_px": None,
                    "pipeline_status": "ok",
                }
                with np.load(case_dir / f"pair_{source_index + 1}_to_3.npz") as data:
                    try:
                        result = estimate_homography(
                            data["source_points"],
                            data["destination_points"],
                            case["threshold_px"],
                            permutation_seed=seed,
                            confidence=config["confidence"],
                            max_iters=config["max_iters"],
                        )
                        metrics = pair_metrics(
                            data["source_points"],
                            data["destination_points"],
                            result["H"],
                            result["inlier_mask"],
                            images[source_index].shape,
                            images[ANCHOR_INDEX].shape,
                        )
                        for key in ("inliers", "inlier_ratio_pct", "rmse_px"):
                            row[key] = metrics[key]
                        corners = _corners(images[source_index].shape).reshape(-1, 1, 2)
                        baseline = cv2.perspectiveTransform(corners, data["H"])
                        perturbed = cv2.perspectiveTransform(corners, result["H"])
                        deviations = np.linalg.norm(perturbed - baseline, axis=2)
                        row["corner_rmse_px"] = float(np.sqrt(np.mean(np.square(deviations))))
                        row["corner_max_deviation_px"] = float(deviations.max())
                        matrices[source_index] = result["H"]
                        for name, value in result.items():
                            arrays[f"pair_{source_index + 1}_{name}"] = value
                    except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
                        row.update(status="error", error=str(error))
                seed_rows.append(row)
            pipeline_status = "ok"
            if any(row["status"] != "ok" for row in seed_rows):
                pipeline_status = "error"
            else:
                try:
                    canvas_geometry(
                        [image.shape for image in images], matrices, config["max_canvas_pixels"]
                    )
                except ValueError:
                    pipeline_status = "error"
            for row in seed_rows:
                row["pipeline_status"] = pipeline_status
            rows.extend(seed_rows)
            np.savez_compressed(directory / f"{method}_seed_{seed}.npz", **arrays)
    return rows


def finalize_selection(run_dir: Path, choices: dict, repeats: int = 5, warmup: int = 1) -> dict:
    """Đo lại các lựa chọn đã xem ảnh; không tự suy ra chất lượng từ RMSE."""
    run_dir = Path(run_dir)
    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("Số lần đo/warmup không hợp lệ.")
    validation = validate_run(run_dir)
    if not validation["valid"]:
        raise ValueError("Run đầu vào chưa hợp lệ: " + "; ".join(validation["errors"][:3]))
    manifest = json.loads((run_dir / "run.json").read_text())
    config = manifest["config"]
    metrics = {row["case_id"]: row for row in read_csv(run_dir / "metrics.csv")}
    if set(choices) != set(config["methods"]):
        raise ValueError("Cần một lựa chọn cho mỗi phương pháp trong run.")
    benchmark_ids = []
    for method, choice in choices.items():
        case_id = choice.get("case_id")
        if (
            case_id not in metrics
            or metrics[case_id]["method"] != method
            or metrics[case_id]["status"] != "ok"
        ):
            raise ValueError(f"Lựa chọn không phải cấu hình thành công của {method}.")
        if not choice.get("reason", "").strip():
            raise ValueError("Cần ghi lý do chọn sau khi xem hình.")
        baseline_id = _case_id(method, 0.75, 3.0)
        for selected_id in (baseline_id, case_id):
            if (
                selected_id in metrics
                and metrics[selected_id]["status"] == "ok"
                and selected_id not in benchmark_ids
            ):
                benchmark_ids.append(selected_id)
    images, _ = load_sequence(run_dir / "inputs")
    shutil.copyfile(Path(__file__), run_dir / "finalization_source.py")
    manifest["finalization_source_sha256"] = _digest(run_dir / "finalization_source.py")
    samples, summaries = [], []
    previous_threads, previous_opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        for case_id in benchmark_ids:
            case = json.loads((run_dir / "cases" / case_id / "case.json").read_text())
            print(f"Benchmark {case_id}: warmup={warmup}, repeats={repeats}", flush=True)
            measured = []
            for index in range(-warmup, repeats):
                result = run_pipeline(
                    images, case["method"], case["ratio"], case["threshold_px"], config
                )
                sample = {
                    "case_id": case_id,
                    "repeat": index,
                    "warmup": index < 0,
                    "status": result["status"],
                    "error": result["error"],
                    "elapsed_ms": result.get("total_stitching_ms"),
                    "elapsed_to_failure_ms": result.get("elapsed_to_failure_ms"),
                    "scope": TIMING_SCOPE,
                }
                samples.append(sample)
                if index >= 0 and result["status"] == "ok":
                    measured.append(result["total_stitching_ms"])
                del result
            all_ok = all(
                sample["status"] == "ok" for sample in samples if sample["case_id"] == case_id
            )
            summaries.append(
                {
                    "case_id": case_id,
                    "method": case["method"],
                    "n_samples": len(measured),
                    "requested_samples": repeats,
                    "warmup": warmup,
                    "status": "ok" if all_ok else "error",
                    "median_ms": float(np.median(measured)) if all_ok else None,
                    "iqr_ms": (
                        float(np.subtract(*np.percentile(measured, [75, 25]))) if all_ok else None
                    ),
                    "scope": TIMING_SCOPE,
                    "timing_kind": "median_iqr",
                }
            )
        robustness_rows = _robustness(run_dir, choices, images, config)
        _write_csv(run_dir / "timings.csv", samples)
        _write_csv(run_dir / "benchmarks.csv", summaries)
        _write_csv(run_dir / "robustness.csv", robustness_rows)
        comparison = []
        timing_by_id = {row["case_id"]: row for row in summaries}
        for method, choice in choices.items():
            baseline_id = _case_id(method, 0.75, 3.0)
            for case_id in dict.fromkeys([baseline_id, choice["case_id"]]):
                if case_id not in metrics:
                    continue
                role = (
                    "baseline_and_selected"
                    if case_id == baseline_id == choice["case_id"]
                    else "baseline" if case_id == baseline_id else "selected"
                )
                row = dict(metrics[case_id], comparison_role=role)
                timing = timing_by_id.get(case_id, {})
                row.update(
                    benchmark_status=timing.get("status", "not_measured"),
                    median_ms=timing.get("median_ms"),
                    iqr_ms=timing.get("iqr_ms"),
                    n_samples=timing.get("n_samples", 0),
                )
                comparison.append(row)
        _write_csv(run_dir / "comparison.csv", comparison)
        selection = {
            "run_id": manifest["run_id"],
            "status": "reviewed",
            "choices": choices,
            "scope": "Selected within this image sequence and measured parameter grid; visual review is not ground truth.",
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            "permutation_note": "Input-order sensitivity; the main H and inlier mask remain unchanged.",
            "benchmark_case_ids": benchmark_ids,
        }
        _write_json(run_dir / "selection.json", selection)
        manifest["selection_status"] = "reviewed"
        _write_json(run_dir / "run.json", manifest)
        build_figures(run_dir)
        return _refresh_validation(run_dir)
    finally:
        cv2.setNumThreads(previous_threads)
        cv2.ocl.setUseOpenCL(previous_opencl)


def analyze_sensitivity(run_dir: Path) -> list[dict]:
    """Giữ mọi cấu hình khi khảo sát ngoại suy góc ảnh, không chọn permutation tốt nhất."""
    run_dir = Path(run_dir)
    checked = validate_run(run_dir)
    if not checked["valid"]:
        raise ValueError(str(checked["errors"]))
    manifest = json.loads((run_dir / "run.json").read_text())
    images, _ = load_sequence(run_dir / "inputs")
    rows, summaries = [], []
    previous_threads, previous_opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        for case in read_csv(run_dir / "metrics.csv"):
            result = []
            if case["status"] == "ok":
                result = _robustness(
                    run_dir,
                    {case["method"]: {"case_id": case["case_id"]}},
                    images,
                    manifest["config"],
                    directory=run_dir / "sensitivity_search" / case["case_id"],
                )
            valid = [row for row in result if row["status"] == "ok"]
            rows.extend(result)
            summaries.append(
                {
                    "case_id": case["case_id"],
                    "method": case["method"],
                    "successful_pairs": len(valid),
                    "expected_pairs": 9,
                    "successful_pipelines": (
                        sum(
                            all(
                                row["pipeline_status"] == "ok"
                                for row in result
                                if row["permutation_seed"] == seed
                            )
                            for seed in (42, 123, 2026)
                        )
                        if result
                        else 0
                    ),
                    "max_corner_deviation_px": max(
                        (row["corner_max_deviation_px"] for row in valid), default=None
                    ),
                    "main_rmse_px": case["rmse_px"],
                    "main_inliers": case["inliers"],
                }
            )
    finally:
        cv2.setNumThreads(previous_threads)
        cv2.ocl.setUseOpenCL(previous_opencl)
    _write_csv(run_dir / "sensitivity_search.csv", rows)
    _write_csv(run_dir / "sensitivity_search_summary.csv", summaries)
    _refresh_validation(run_dir)
    return summaries


def compare_blending(run_dir: Path, repeats: int = 5, warmup: int = 1) -> list[dict]:
    """So sánh feather và 3/5/7 band trên cùng H; timing gồm dựng canvas, warp, blend và crop."""
    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("Số lần đo/warmup không hợp lệ.")
    run_dir = Path(run_dir)
    checked = validate_run(run_dir)
    if not checked["valid"]:
        raise ValueError(str(checked["errors"]))
    selection = json.loads((run_dir / "selection.json").read_text())
    if selection["status"] != "reviewed":
        raise ValueError("Cần chọn cấu hình sau khi xem panorama trước khi so sánh blending.")
    manifest = json.loads((run_dir / "run.json").read_text())
    images, _ = load_sequence(run_dir / "inputs")
    directory = run_dir / "blending"
    directory.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__), directory / "source_code.py")
    samples, rows = [], []
    previous_threads, previous_opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        for method, choice in selection["choices"].items():
            case_dir = run_dir / "cases" / choice["case_id"]
            geometry_path = case_dir / "geometry.json"
            original = json.loads(geometry_path.read_text())
            transforms = [np.asarray(H) for H in original["transforms"]]
            reference = cv2.imread(str(case_dir / "panorama.png"))
            reference_mask = cv2.imread(str(case_dir / "mask.png"), 0) > 0
            for bands in (0, 3, 5, 7):
                name = "feather" if bands == 0 else f"multiband_{bands}"
                output_dir = directory / f"{method}_{name}"
                output_dir.mkdir(exist_ok=True)
                times = []
                for index in range(-warmup, repeats):
                    start = time.perf_counter_ns()
                    if bands:
                        panorama, mask, geometry = blend_multiband(
                            images, transforms, manifest["config"]["max_canvas_pixels"], bands
                        )
                    else:
                        panorama, mask, geometry = blend_images(
                            images, transforms, manifest["config"]["max_canvas_pixels"]
                        )
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    samples.append(
                        {
                            "method": method,
                            "blend": name,
                            "repeat": index,
                            "warmup": index < 0,
                            "elapsed_ms": elapsed,
                            "scope": "canvas_setup_warp_blend_and_crop",
                        }
                    )
                    if index >= 0:
                        times.append(elapsed)
                    if index == 0:
                        if (
                            not np.array_equal(mask, reference_mask)
                            or geometry["crop_offset"] != original["crop_offset"]
                        ):
                            raise ValueError("So sánh blending không cùng hỗ trợ hình học.")
                        _save_image(output_dir / "panorama.png", panorama)
                        _save_image(output_dir / "mask.png", mask.astype(np.uint8) * 255)
                        _write_json(
                            output_dir / "geometry.json",
                            dict(
                                geometry,
                                transforms=original["transforms"],
                                source_geometry_sha256=_digest(geometry_path),
                            ),
                        )
                        difference = float(
                            np.mean(np.abs(panorama[mask].astype(np.float32) - reference[mask]))
                        )
                rows.append(
                    {
                        "method": method,
                        "case_id": choice["case_id"],
                        "blend": name,
                        "bands": bands,
                        "n_samples": len(times),
                        "warmup": warmup,
                        "median_ms": float(np.median(times)),
                        "iqr_ms": float(np.subtract(*np.percentile(times, [75, 25]))),
                        "mean_abs_change_from_feather": difference,
                        "scope": "canvas_setup_warp_blend_and_crop",
                        "quality_note": "Pixel change is not a quality score; inspect shared crops.",
                    }
                )
                print(f"Blending {method}/{name}: {rows[-1]['median_ms']:.1f} ms", flush=True)
        _write_csv(run_dir / "blending.csv", rows)
        _write_csv(directory / "timings.csv", samples)
        _build_blending_figures(run_dir)
        _refresh_validation(run_dir)
    finally:
        cv2.setNumThreads(previous_threads)
        cv2.ocl.setUseOpenCL(previous_opencl)
    return rows


def _build_blending_figures(run_dir: Path) -> None:
    selection = json.loads((run_dir / "selection.json").read_text())
    roi_pixels = json.loads((run_dir / "figures/figures.json").read_text())[
        "roi_boxes_anchor_pixels"
    ]
    directory = run_dir / "blending"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for roi, box in dict(panorama=None, **roi_pixels).items():
        figure, axes = plt.subplots(
            len(selection["choices"]), 4, figsize=(18, 4 * len(selection["choices"])), squeeze=False
        )
        for method_index, method in enumerate(selection["choices"]):
            for column, name in enumerate(("feather", "multiband_3", "multiband_5", "multiband_7")):
                output_dir = directory / f"{method}_{name}"
                panorama = cv2.imread(str(output_dir / "panorama.png"))
                geometry = json.loads((output_dir / "geometry.json").read_text())
                shown = panorama if box is None else _anchor_crop(panorama, geometry, box)
                axes[method_index, column].imshow(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
                axes[method_index, column].set_title(f"{method.upper()} / {name}")
                axes[method_index, column].axis("off")
        figure.suptitle(f"Fixed homographies — {roi}")
        figure.tight_layout()
        figure.savefig(directory / f"comparison_{roi}.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


def save_observations(run_dir: Path, observations: dict) -> dict:
    """Lưu ghi nhận thực nghiệm sau khi xem hình, gắn với run đã kiểm tra."""
    run_dir = Path(run_dir)
    validation = validate_run(run_dir)
    if not validation["valid"]:
        raise ValueError(str(validation["errors"]))
    qualitative = observations.get("qualitative", {})
    if any(
        not isinstance(qualitative.get(key), str) or not qualitative[key].strip()
        for key in ("alignment", "seam", "ghosting", "illumination", "perspective")
    ):
        raise ValueError("Cần ghi nhận đủ năm khía cạnh chất lượng panorama.")
    manifest = json.loads((run_dir / "run.json").read_text())
    _write_json(
        run_dir / "analysis.json",
        dict(
            observations,
            run_id=manifest["run_id"],
            reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    return _refresh_validation(run_dir)


def validate_run(run_dir: Path) -> dict:
    """Đối chiếu hash và tính lại số liệu từ correspondence; không chứng nhận chất lượng thị giác."""
    run_dir = Path(run_dir)
    errors = []
    checked_pairs = 0
    try:
        manifest = json.loads((run_dir / "run.json").read_text())
        recorded = manifest.get("artifacts", {})
        if not recorded or set(recorded) != set(_inventory(run_dir)):
            errors.append("Danh mục hash thiếu artifact hoặc chứa đường dẫn không còn tồn tại.")
        for relative, expected in manifest.get("artifacts", {}).items():
            path = run_dir / relative
            if not path.is_file() or _digest(path) != expected:
                errors.append(f"Hash artifact không khớp: {relative}")
        if _digest(run_dir / "source_code.py") != manifest["source_sha256"]:
            errors.append("Source snapshot không khớp.")
        if _digest(run_dir / "inputs/metadata.json") != manifest["input_metadata_sha256"]:
            errors.append("Metadata snapshot không khớp.")
        if (
            "finalization_source_sha256" in manifest
            and _digest(run_dir / "finalization_source.py")
            != manifest["finalization_source_sha256"]
        ):
            errors.append("Source hậu xử lý không khớp.")
        images, _ = load_sequence(run_dir / "inputs")
        rows = read_csv(run_dir / "metrics.csv")
        pair_rows = read_csv(run_dir / "pairs.csv")
        expected_cases = _cases(manifest["config"])
        if len(rows) != len(expected_cases) or {row["case_id"] for row in rows} != {
            case["case_id"] for case in expected_cases
        }:
            errors.append("Số lượng/ID cấu hình không khớp ma trận.")
        shapes = [image.shape for image in images]
        for row in rows:
            case_dir = run_dir / "cases" / row["case_id"]
            stored = json.loads((case_dir / "case.json").read_text())
            for field in METRIC_FIELDS:
                expected = "" if stored[field] is None else str(stored[field])
                if row[field] != expected:
                    errors.append(f"CSV và case.json khác nhau: {row['case_id']}/{field}")
            actual_pairs = []
            detected, used = 0, 0
            for image_index in range(4):
                feature_path = run_dir / "features" / f"{row['method']}_{image_index + 1}.npz"
                if feature_path.exists():
                    with np.load(feature_path) as features:
                        detected += int(features["detected_count"])
                        used += len(features["points"])
                        if len(features["points"]) != len(features["descriptors"]):
                            errors.append(f"Số descriptor không khớp keypoint: {feature_path.name}")
            if detected != int(row["detected_keypoints"]) or used != int(row["used_keypoints"]):
                errors.append(f"Tổng keypoint không khớp: {row['case_id']}")
            available_pairs = [pair for pair in pair_rows if pair["case_id"] == row["case_id"]]
            for pair in available_pairs:
                source_index = int(pair["source_image"]) - 1
                with np.load(case_dir / f"pair_{source_index + 1}_to_3.npz") as data:
                    if len(data["knn"]) != int(pair["initial_matches"]) or len(
                        data["source_points"]
                    ) != int(pair["ratio_pass_matches"]):
                        errors.append(f"Số match sai: {row['case_id']}")
                    if pair["status"] != "ok":
                        continue
                    recomputed = pair_metrics(
                        data["source_points"],
                        data["destination_points"],
                        data["H"],
                        data["inlier_mask"],
                        shapes[source_index],
                        shapes[ANCHOR_INDEX],
                    )
                    recomputed["initial_matches"] = len(data["knn"])
                    actual_pairs.append(recomputed)
                    for key, value in recomputed.items():
                        if not math.isclose(float(pair[key]), value, rel_tol=1e-7, abs_tol=1e-7):
                            errors.append(f"Metric cặp sai: {row['case_id']}/{key}")
                    for image_index, field, column in (
                        (source_index, "source_points", 0),
                        (ANCHOR_INDEX, "destination_points", 1),
                    ):
                        with np.load(
                            run_dir / "features" / f"{row['method']}_{image_index + 1}.npz"
                        ) as features:
                            if len(features["points"]) > manifest["config"]["max_features"]:
                                errors.append("Vượt ngân sách đặc trưng.")
                            indices = data["match_indices"][:, column]
                            if not np.array_equal(features["points"][indices], data[field]):
                                errors.append(
                                    f"Correspondence không khớp keypoint: {row['case_id']}"
                                )
                    checked_pairs += 1
            if row["status"] == "ok":
                if (
                    len(actual_pairs) != 3
                    or {int(pair["source_image"]) for pair in available_pairs} != {1, 2, 4}
                    or any(int(pair["destination_image"]) != 3 for pair in available_pairs)
                ):
                    errors.append(f"Thiếu kết nối trong panorama: {row['case_id']}")
                    continue
                for key, value in summarize_pairs(actual_pairs).items():
                    if not math.isclose(float(row[key]), value, rel_tol=1e-7, abs_tol=1e-7):
                        errors.append(f"Metric tổng hợp sai: {row['case_id']}/{key}")
                geometry = json.loads((case_dir / "geometry.json").read_text())
                canvas = canvas_geometry(
                    shapes,
                    [np.asarray(H) for H in geometry["transforms"]],
                    manifest["config"]["max_canvas_pixels"],
                )
                if canvas["canvas_size"] != geometry["canvas_size"]:
                    errors.append(f"Canvas không khớp: {row['case_id']}")
                panorama = cv2.imread(str(case_dir / "panorama.png"))
                mask = cv2.imread(str(case_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
                if (
                    panorama is None
                    or mask is None
                    or panorama.shape[:2] != mask.shape
                    or list(panorama.shape[1::-1]) != geometry["output_size"]
                    or not mask.any()
                ):
                    errors.append(f"Panorama/mask không hợp lệ: {row['case_id']}")
                if (
                    not math.isfinite(float(row["total_stitching_ms"]))
                    or float(row["total_stitching_ms"]) <= 0
                ):
                    errors.append(f"Timing không hợp lệ: {row['case_id']}")
            elif (
                row["total_stitching_ms"]
                or not row["error"]
                or float(row["elapsed_to_failure_ms"]) <= 0
            ):
                errors.append(f"Bản ghi lỗi không hợp lệ: {row['case_id']}")
        selection = json.loads((run_dir / "selection.json").read_text())
        if selection["run_id"] != manifest["run_id"]:
            errors.append("Selection thuộc run khác.")
        if selection["status"] == "reviewed":
            if set(selection["choices"]) != set(manifest["config"]["methods"]):
                errors.append("Selection thiếu hoặc thừa phương pháp.")
            benchmarks = read_csv(run_dir / "benchmarks.csv")
            samples = read_csv(run_dir / "timings.csv")
            required_benchmarks = set()
            valid_ids = {row["case_id"] for row in rows if row["status"] == "ok"}
            for method, choice in selection["choices"].items():
                required_benchmarks.add(choice["case_id"])
                baseline_id = _case_id(method, 0.75, 3.0)
                if baseline_id in valid_ids:
                    required_benchmarks.add(baseline_id)
            if {row["case_id"] for row in benchmarks} != required_benchmarks or len(
                benchmarks
            ) != len(required_benchmarks):
                errors.append("Danh sách benchmark không khớp baseline/selection.")
            for benchmark in benchmarks:
                measured = [
                    float(sample["elapsed_ms"])
                    for sample in samples
                    if sample["case_id"] == benchmark["case_id"]
                    and sample["warmup"] == "False"
                    and sample["status"] == "ok"
                ]
                if len(measured) != int(benchmark["n_samples"]):
                    errors.append("Số mẫu timing không khớp.")
                if benchmark["status"] == "ok":
                    if len(measured) != int(benchmark["requested_samples"]) or not math.isclose(
                        float(benchmark["median_ms"]), float(np.median(measured))
                    ):
                        errors.append("Median benchmark không khớp mẫu thô.")
                    if not math.isclose(
                        float(benchmark["iqr_ms"]),
                        float(np.subtract(*np.percentile(measured, [75, 25]))),
                        abs_tol=1e-8,
                    ):
                        errors.append("IQR benchmark không khớp mẫu thô.")
            for method, choice in selection["choices"].items():
                matching = [
                    row
                    for row in rows
                    if row["case_id"] == choice["case_id"]
                    and row["method"] == method
                    and row["status"] == "ok"
                ]
                if len(matching) != 1:
                    errors.append("Selection không trỏ đến cấu hình hợp lệ.")
            robustness = read_csv(run_dir / "robustness.csv")
            if len(robustness) != len(selection["choices"]) * 9:
                errors.append("Thiếu bản ghi kiểm tra thứ tự điểm.")
            for method, choice in selection["choices"].items():
                for seed in (42, 123, 2026):
                    seed_rows = [
                        row
                        for row in robustness
                        if row["method"] == method and int(row["permutation_seed"]) == seed
                    ]
                    if len(seed_rows) != 3 or {int(row["source_image"]) for row in seed_rows} != {
                        1,
                        2,
                        4,
                    }:
                        errors.append(f"Kiểm tra thứ tự điểm không đủ ba kết nối: {method}/{seed}")
                        continue
                    matrices = [np.eye(3) for _ in images]
                    pipeline_ok = all(row["status"] == "ok" for row in seed_rows)
                    with np.load(run_dir / "robustness" / f"{method}_seed_{seed}.npz") as perturbed:
                        for row in seed_rows:
                            if row["status"] != "ok":
                                continue
                            index = int(row["source_image"]) - 1
                            prefix = f"pair_{index + 1}_"
                            with np.load(
                                run_dir / "cases" / choice["case_id"] / f"pair_{index + 1}_to_3.npz"
                            ) as base:
                                permutation = perturbed[prefix + "permutation"]
                                expected = np.random.default_rng(seed).permutation(
                                    len(base["source_points"])
                                )
                                if not np.array_equal(permutation, expected):
                                    errors.append("Permutation không khớp seed đã ghi.")
                                H = perturbed[prefix + "H"]
                                matrices[index] = H
                                metric = pair_metrics(
                                    base["source_points"],
                                    base["destination_points"],
                                    H,
                                    perturbed[prefix + "inlier_mask"],
                                    shapes[index],
                                    shapes[ANCHOR_INDEX],
                                )
                                for key in ("inliers", "inlier_ratio_pct", "rmse_px"):
                                    if not math.isclose(
                                        float(row[key]), metric[key], rel_tol=1e-7, abs_tol=1e-7
                                    ):
                                        errors.append(
                                            f"Metric robustness không khớp: {method}/{seed}/{key}"
                                        )
                                corners = _corners(shapes[index]).reshape(-1, 1, 2)
                                deltas = cv2.perspectiveTransform(
                                    corners, H
                                ) - cv2.perspectiveTransform(corners, base["H"])
                                distances = np.linalg.norm(deltas, axis=2)
                                corner_values = {
                                    "corner_rmse_px": float(np.sqrt(np.mean(distances**2))),
                                    "corner_max_deviation_px": float(distances.max()),
                                }
                                for key, value in corner_values.items():
                                    if not math.isclose(
                                        float(row[key]), value, rel_tol=1e-7, abs_tol=1e-7
                                    ):
                                        errors.append("Sai độ lệch góc ảnh trong robustness.")
                    if pipeline_ok:
                        try:
                            canvas_geometry(
                                shapes, matrices, manifest["config"]["max_canvas_pixels"]
                            )
                        except ValueError:
                            pipeline_ok = False
                    expected_status = "ok" if pipeline_ok else "error"
                    if any(row["pipeline_status"] != expected_status for row in seed_rows):
                        errors.append("Trạng thái robustness không khớp hình học.")
        if (run_dir / "analysis.json").exists():
            notes = json.loads((run_dir / "analysis.json").read_text())
            if notes.get("run_id") != manifest["run_id"]:
                errors.append("Ghi nhận thực nghiệm thuộc run khác.")
            if any(
                not isinstance(notes.get("qualitative", {}).get(key), str)
                or not notes["qualitative"][key].strip()
                for key in ("alignment", "seam", "ghosting", "illumination", "perspective")
            ):
                errors.append("Ghi nhận thực nghiệm thiếu khía cạnh chất lượng.")
        if (run_dir / "blending.csv").exists():
            blends = read_csv(run_dir / "blending.csv")
            raw = read_csv(run_dir / "blending/timings.csv")
            expected_blends = {
                (method, name)
                for method in selection["choices"]
                for name in ("feather", "multiband_3", "multiband_5", "multiband_7")
            }
            if (
                len(blends) != len(expected_blends)
                or {(row["method"], row["blend"]) for row in blends} != expected_blends
            ):
                errors.append("Thiếu cấu hình blending so sánh.")
            for row in blends:
                directory = run_dir / "blending" / f"{row['method']}_{row['blend']}"
                base_dir = run_dir / "cases" / selection["choices"][row["method"]]["case_id"]
                geometry = json.loads((directory / "geometry.json").read_text())
                base_geometry = json.loads((base_dir / "geometry.json").read_text())
                if (
                    geometry["source_geometry_sha256"] != _digest(base_dir / "geometry.json")
                    or geometry["transforms"] != base_geometry["transforms"]
                    or geometry["crop_offset"] != base_geometry["crop_offset"]
                ):
                    errors.append("Blending đã thay đổi homography hoặc crop.")
                mask = cv2.imread(str(directory / "mask.png"), 0)
                base_mask = cv2.imread(str(base_dir / "mask.png"), 0)
                output = cv2.imread(str(directory / "panorama.png"))
                reference = cv2.imread(str(base_dir / "panorama.png"))
                if not np.array_equal(mask, base_mask) or output.shape != reference.shape:
                    errors.append("Mask/kích thước blending không cùng geometry.")
                else:
                    difference = float(
                        np.mean(np.abs(output[mask > 0].astype(np.float32) - reference[mask > 0]))
                    )
                    if not math.isclose(
                        difference, float(row["mean_abs_change_from_feather"]), abs_tol=1e-6
                    ):
                        errors.append("Độ thay đổi pixel blending không khớp.")
                samples = [
                    float(sample["elapsed_ms"])
                    for sample in raw
                    if sample["method"] == row["method"]
                    and sample["blend"] == row["blend"]
                    and sample["warmup"] == "False"
                ]
                if (
                    len(samples) != int(row["n_samples"])
                    or not samples
                    or not math.isclose(float(row["median_ms"]), float(np.median(samples)))
                    or not math.isclose(
                        float(row["iqr_ms"]),
                        float(np.subtract(*np.percentile(samples, [75, 25]))),
                        abs_tol=1e-7,
                    )
                ):
                    errors.append("Thống kê timing blending không khớp mẫu thô.")
        if (run_dir / "sensitivity_search_summary.csv").exists():
            summaries = read_csv(run_dir / "sensitivity_search_summary.csv")
            raw = read_csv(run_dir / "sensitivity_search.csv")
            if len(summaries) != len(rows) or {item["case_id"] for item in summaries} != {
                item["case_id"] for item in rows
            }:
                errors.append("Khảo sát sensitivity thiếu cấu hình.")
            for item in summaries:
                matching = [entry for entry in raw if entry["case_id"] == item["case_id"]]
                valid = [entry for entry in matching if entry["status"] == "ok"]
                if len(valid) != int(item["successful_pairs"]):
                    errors.append("Số cặp sensitivity không khớp.")
                if valid and not math.isclose(
                    max(float(entry["corner_max_deviation_px"]) for entry in valid),
                    float(item["max_corner_deviation_px"]),
                ):
                    errors.append("Độ lệch cực đại sensitivity không khớp.")
    except (OSError, ValueError, KeyError, IndexError, TypeError, cv2.error) as error:
        errors.append(f"{type(error).__name__}: {error}")
    return {
        "valid": not errors,
        "errors": errors,
        "checked_pairs": checked_pairs,
        "scope": "Artifact integrity and recomputed geometric metrics; not visual-quality or ground-truth certification.",
    }


ROI_BOXES = {
    "sign": [0.31, 0.31, 0.62, 0.62],
    "column": [0.08, 0.08, 0.29, 0.97],
    "lantern": [0.62, 0.22, 0.71, 0.55],
    "overlap_detail": [0.52, 0.58, 0.90, 0.96],
}


def _roi_pixels(shape) -> dict:
    height, width = shape[:2]
    return {
        name: [
            round(box[0] * width),
            round(box[1] * height),
            round(box[2] * width),
            round(box[3] * height),
        ]
        for name, box in ROI_BOXES.items()
    }


def _anchor_crop(panorama, geometry, box):
    translation = np.asarray(geometry["translation"])
    offset = np.asarray(geometry["crop_offset"])
    shift_x, shift_y = np.rint(translation[:2, 2] - offset).astype(int)
    x0, y0, x1, y1 = box
    crop = panorama[
        max(0, y0 + shift_y) : min(panorama.shape[0], y1 + shift_y),
        max(0, x0 + shift_x) : min(panorama.shape[1], x1 + shift_x),
    ]
    if crop.size == 0:
        raise ValueError("Crop trong hệ ảnh tham chiếu bị rỗng.")
    return crop


def build_figures(run_dir: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text())
    rows = read_csv(run_dir / "metrics.csv")
    images, metadata = load_sequence(run_dir / "inputs")
    directory = run_dir / "figures"
    directory.mkdir(exist_ok=True)
    paths = {}
    previous_figures = directory / "figures.json"
    roi_pixels = (
        json.loads(previous_figures.read_text())["roi_boxes_anchor_pixels"]
        if previous_figures.exists()
        else _roi_pixels(images[ANCHOR_INDEX].shape)
    )
    selection = json.loads((run_dir / "selection.json").read_text())

    def save_figure(name, figure):
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        paths[name] = path

    for method in manifest["config"]["methods"]:
        method_rows = [row for row in rows if row["method"] == method]
        successful = [row for row in method_rows if row["status"] == "ok"]
        for index, image in enumerate(images):
            feature_path = run_dir / "features" / f"{method}_{index + 1}.npz"
            if not feature_path.exists():
                continue
            with np.load(feature_path) as features:
                keypoints = [
                    cv2.KeyPoint(float(x), float(y), 5.0) for x, y in features["points"][:500]
                ]
                canvas = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
                figure, axis = plt.subplots(figsize=(10, 5))
                axis.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                axis.set_title(
                    f"{method.upper()} — image {index + 1}: detected {int(features['detected_count'])}, used {len(features['points'])}; first 500 shown"
                )
                axis.axis("off")
                save_figure(f"{method}_keypoints_{index + 1}", figure)
        if not successful:
            continue
        ratios, thresholds = manifest["config"]["ratios"], manifest["config"]["thresholds_px"]
        figure, axes = plt.subplots(
            len(ratios),
            len(thresholds),
            squeeze=False,
            figsize=(5 * len(thresholds), 2.8 * len(ratios)),
        )
        crop_figures = {
            name: plt.subplots(
                len(ratios),
                len(thresholds),
                squeeze=False,
                figsize=(5 * len(thresholds), 3.6 * len(ratios)),
            )
            for name in roi_pixels
        }
        for row in method_rows:
            i, j = ratios.index(float(row["ratio"])), thresholds.index(float(row["threshold_px"]))
            axis = axes[i, j]
            title = f"ratio={row['ratio']}, RANSAC={row['threshold_px']} px"
            if row["status"] == "ok":
                case_dir = run_dir / "cases" / row["case_id"]
                panorama = cv2.imread(str(case_dir / "panorama.png"))
                geometry = json.loads((case_dir / "geometry.json").read_text())
                thumbnail = cv2.resize(
                    panorama,
                    (900, max(1, round(900 * panorama.shape[0] / panorama.shape[1]))),
                    interpolation=cv2.INTER_AREA,
                )
                axis.imshow(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
                for name, box in roi_pixels.items():
                    crop_axis = crop_figures[name][1][i, j]
                    crop_axis.imshow(
                        cv2.cvtColor(_anchor_crop(panorama, geometry, box), cv2.COLOR_BGR2RGB)
                    )
                    crop_axis.set_title(title, fontsize=9)
                    crop_axis.axis("off")
                title += f"\nRMSE={float(row['rmse_px']):.3f} px, inliers={row['inliers']}"
                del panorama
            else:
                axis.text(
                    0.5,
                    0.5,
                    row["error"],
                    ha="center",
                    va="center",
                    wrap=True,
                    transform=axis.transAxes,
                )
                for _, crop_axes in crop_figures.values():
                    crop_axes[i, j].text(0.5, 0.5, "FAILED", ha="center")
                    crop_axes[i, j].axis("off")
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        figure.suptitle(f"{method.upper()} — full parameter grid; common anchor: image 3")
        figure.tight_layout()
        save_figure(f"{method}_panorama_grid", figure)
        for name, (crop_figure, _) in crop_figures.items():
            crop_figure.suptitle(f"{method.upper()} — {name}, fixed crop in image-3 coordinates")
            crop_figure.tight_layout()
            save_figure(f"{method}_crop_grid_{name}", crop_figure)
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ratio in ratios:
            values = sorted(
                [row for row in successful if float(row["ratio"]) == ratio],
                key=lambda row: float(row["threshold_px"]),
            )
            x = [float(row["threshold_px"]) for row in values]
            axes[0].plot(
                x, [float(row["inlier_ratio_pct"]) for row in values], "o-", label=f"ratio {ratio}"
            )
            axes[1].plot(x, [float(row["rmse_px"]) for row in values], "o-", label=f"ratio {ratio}")
        for axis, label in zip(axes, ("Inlier ratio (%)", "Inlier reprojection RMSE (px)")):
            axis.set_xlabel("RANSAC threshold (px)")
            axis.set_ylabel(label)
            axis.legend()
            axis.grid(alpha=0.3)
        figure.suptitle(f"{method.upper()} — threshold sensitivity (different inlier sets)")
        figure.tight_layout()
        save_figure(f"{method}_sensitivity", figure)
        matching_ids = [_case_id(method, 0.75, 3.0)]
        if method in selection.get("choices", {}):
            matching_ids.append(selection["choices"][method]["case_id"])
        for case_id in dict.fromkeys(matching_ids):
            if not any(row["case_id"] == case_id and row["status"] == "ok" for row in method_rows):
                continue
            for source_index in (0, 1, 3):
                with np.load(run_dir / "features" / f"{method}_{source_index + 1}.npz") as feature:
                    source_keypoints = [
                        cv2.KeyPoint(float(x), float(y), 3.0) for x, y in feature["points"]
                    ]
                with np.load(run_dir / "features" / f"{method}_3.npz") as feature:
                    destination_keypoints = [
                        cv2.KeyPoint(float(x), float(y), 3.0) for x, y in feature["points"]
                    ]
                with np.load(
                    run_dir / "cases" / case_id / f"pair_{source_index + 1}_to_3.npz"
                ) as data:
                    for stage in ("initial", "inlier"):
                        indices = (
                            data["knn"][:, :2].astype(int)
                            if stage == "initial"
                            else data["match_indices"][data["inlier_mask"]]
                        )
                        generator = np.random.default_rng(42)
                        chosen = np.sort(
                            generator.choice(len(indices), min(80, len(indices)), replace=False)
                        )
                        matches = [
                            cv2.DMatch(int(indices[index, 0]), int(indices[index, 1]), 0.0)
                            for index in chosen
                        ]
                        drawn = cv2.drawMatches(
                            images[source_index],
                            source_keypoints,
                            images[ANCHOR_INDEX],
                            destination_keypoints,
                            matches,
                            None,
                            matchColor=(0, 255, 0),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                        )
                        path = directory / f"{case_id}_{source_index + 1}_to_3_{stage}.jpg"
                        _save_image(path, drawn)
                        paths[path.stem] = path
    if selection["status"] == "reviewed":
        methods = list(selection["choices"])
        figure, axes = plt.subplots(2, len(methods), squeeze=False, figsize=(6 * len(methods), 7.5))
        crop_figure, crop_axes = plt.subplots(
            len(roi_pixels),
            len(methods),
            squeeze=False,
            figsize=(5 * len(methods), 3.4 * len(roi_pixels)),
        )
        for column, method in enumerate(methods):
            selected_id = selection["choices"][method]["case_id"]
            for i, (role, case_id) in enumerate(
                (("Baseline", _case_id(method, 0.75, 3.0)), ("Selected", selected_id))
            ):
                path = run_dir / "cases" / case_id / "panorama.png"
                axis = axes[i, column]
                if path.is_file():
                    panorama = cv2.imread(str(path))
                    thumb = cv2.resize(
                        panorama,
                        (1200, round(1200 * panorama.shape[0] / panorama.shape[1])),
                        interpolation=cv2.INTER_AREA,
                    )
                    axis.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
                    if role == "Selected":
                        geometry = json.loads((path.parent / "geometry.json").read_text())
                        for roi_index, (name, box) in enumerate(roi_pixels.items()):
                            crop_axes[roi_index, column].imshow(
                                cv2.cvtColor(
                                    _anchor_crop(panorama, geometry, box), cv2.COLOR_BGR2RGB
                                )
                            )
                            crop_axes[roi_index, column].set_title(f"{method.upper()} — {name}")
                            crop_axes[roi_index, column].axis("off")
                    del panorama
                else:
                    axis.text(0.5, 0.5, "FAILED", ha="center")
                axis.set_title(f"{method.upper()} {role}\n{case_id}", fontsize=10)
                axis.axis("off")
        figure.tight_layout(h_pad=3.0, w_pad=1.0)
        save_figure("baseline_selected_panoramas", figure)
        crop_figure.tight_layout()
        save_figure("selected_crop_comparison", crop_figure)
        comparisons = read_csv(run_dir / "comparison.csv")
        measured = [row for row in comparisons if row["benchmark_status"] == "ok"]
        figure, axis = plt.subplots(figsize=(max(7, len(measured) * 1.7), 4))
        values = [float(row["median_ms"]) for row in measured]
        axis.bar(range(len(values)), values, color="0.4")
        axis.set_xticks(
            range(len(values)),
            [f"{row['method'].upper()}\n{row['comparison_role']}" for row in measured],
            fontsize=8,
        )
        axis.set_ylabel("End-to-end median runtime (ms)")
        axis.set_ylim(0, max(values, default=1) * 1.25)
        for index, row in enumerate(measured):
            axis.text(
                index,
                values[index],
                f"{values[index]:.1f}\nIQR {float(row['iqr_ms']):.1f}\nn={row['n_samples']}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        figure.tight_layout()
        save_figure("runtime_comparison", figure)
    search_path = run_dir / "sensitivity_search_summary.csv"
    if search_path.exists():
        search_rows = {row["case_id"]: row for row in read_csv(search_path)}
        methods = manifest["config"]["methods"]
        ratios = manifest["config"]["ratios"]
        thresholds = manifest["config"]["thresholds_px"]
        finite = [
            float(row["max_corner_deviation_px"])
            for row in search_rows.values()
            if row["max_corner_deviation_px"]
        ]
        if finite:
            figure, axes = plt.subplots(
                1, len(methods), figsize=(5.5 * len(methods), 4.8), squeeze=False
            )
            for axis, method in zip(axes[0], methods):
                matrix = np.full((len(ratios), len(thresholds)), np.nan)
                for i, ratio in enumerate(ratios):
                    for j, threshold in enumerate(thresholds):
                        entry = search_rows.get(_case_id(method, ratio, threshold), {})
                        if entry.get("max_corner_deviation_px"):
                            matrix[i, j] = float(entry["max_corner_deviation_px"])
                plotted = axis.imshow(matrix, cmap="magma", vmin=0, vmax=max(max(finite), 1))
                for i in range(len(ratios)):
                    for j in range(len(thresholds)):
                        label = f"{matrix[i, j]:.2f}" if np.isfinite(matrix[i, j]) else "N/A"
                        axis.text(j, i, label, ha="center", va="center", color="white")
                axis.set_xticks(range(len(thresholds)), thresholds)
                axis.set_yticks(range(len(ratios)), ratios)
                axis.set_xlabel("RANSAC threshold (px)")
                axis.set_ylabel("Lowe ratio")
                axis.set_title(method.upper())
            figure.colorbar(
                plotted,
                ax=axes[0].tolist(),
                label="Max projected-corner deviation (px)",
                shrink=0.8,
            )
            figure.suptitle("Input-order sensitivity; three fixed permutations; main H unchanged")
            save_figure("input_order_sensitivity", figure)
    figure_manifest = {
        "run_id": manifest["run_id"],
        "roi_boxes_anchor_pixels": roi_pixels,
        "anchor_filename": metadata["images"][ANCHOR_INDEX]["filename"],
        "display_seed": 42,
        "dpi": 300,
        "figure_code_sha256": _digest(Path(__file__)),
        "figures": {name: str(path.relative_to(run_dir)) for name, path in paths.items()},
    }
    _write_json(directory / "figures.json", figure_manifest)
    manifest["artifacts"] = _inventory(run_dir)
    _write_json(run_dir / "run.json", manifest)
    return paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Part C: four-image SIFT/ORB/AKAZE stitching")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    config = SMOKE_CONFIG if args.profile == "smoke" else DEFAULT_CONFIG
    run_dir = run_experiments(args.input_dir, args.output, config)
    print(run_dir.resolve())


if __name__ == "__main__":
    main()
