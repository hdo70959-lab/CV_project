"""Tạo nhiễu, khử nhiễu, tăng sắc nét và đo chất lượng ảnh cho Part A."""

from __future__ import annotations

import hashlib
import math
import copy
import csv
import json
import platform
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


DEFAULT_INPUT = Path("data/part_a/succulent_20231108_171619.jpg")
DEFAULT_OUTPUT = Path("results/part_a/official")

DEFAULT_CONFIG = {
    "profile": "full",
    "max_long_edge": 1024,
    "noise_levels": {"gaussian": [10, 20, 30], "salt_pepper": [0.01, 0.03, 0.05]},
    "seeds": [42, 43, 44],
    "save_seeds": [42],
    "denoise_kernels": [3, 5, 7, 9, 11],
    "gaussian_sigmas": [0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6],
    "laplacian_kernels": [3, 5, 7],
    "alphas": [0.1, 0.2, 0.25, 0.5],
    "warmup": 2,
    "repeats": 10,
    "roi_boxes": {
        "pot_wall": [0.29, 0.75, 0.64, 0.85],
        "leaf_edges": [0.25, 0.39, 0.53, 0.59],
        "gravel": [0.28, 0.6, 0.5, 0.67],
    },
    "selected_denoisers": {
        "gaussian": {
            "10.0": {"mean": [3, None], "gaussian": [5, 0.8], "median": [3, None]},
            "20.0": {"mean": [5, None], "gaussian": [7, 1.2], "median": [5, None]},
            "30.0": {"mean": [5, None], "gaussian": [9, 1.5], "median": [7, None]},
        },
        "salt_pepper": {
            "0.01": {"mean": [3, None], "gaussian": [5, 1.0], "median": [3, None]},
            "0.03": {"mean": [5, None], "gaussian": [7, 1.4], "median": [3, None]},
            "0.05": {"mean": [5, None], "gaussian": [11, 1.6], "median": [3, None]},
        },
    },
}
# Giữ cấu hình cũ để kiểm chứng các run chưa lưu selected_denoisers.
LEGACY_GAUSSIAN_SETTINGS = {
    "gaussian": {10: (3, 0.6), 20: (3, 0.8), 30: (5, 0.8)},
    "salt_pepper": {0.01: (3, 0.6), 0.03: (5, 0.8), 0.05: (7, 1.2)},
}
RECOMMENDED_SHARPENING = {"laplacian_kernel_size": 3, "alpha": 0.1}


SMOKE_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
SMOKE_CONFIG.update(
    profile="smoke", max_long_edge=256,
    noise_levels={"gaussian": [20], "salt_pepper": [0.03]},
    seeds=[42], denoise_kernels=[3], gaussian_sigmas=[1.2],
    laplacian_kernels=[3], alphas=[0.25], warmup=1, repeats=2,
)
SMOKE_CONFIG.pop("selected_denoisers", None)

CASE_FIELDS = ["case_id", "noise_type", "noise_level", "seed", "stage", "method",
    "denoise_kernel_size", "filter_sigma", "laplacian_kernel_size", "laplacian_scale", "alpha"]
METRIC_FIELDS = CASE_FIELDS + ["mse", "psnr_db", "psnr_is_infinite", "ssim",
    "runtime_ms", "runtime_iqr_ms", "runtime_reason", "status", "error",
    "image_saved", "image_path", "image_sha256", "image_reason", "measurement_run_id"]
TIMING_FIELDS = ["case_id", "repeat", "elapsed_ms", "scope"]
GROUP_FIELDS = [field for field in CASE_FIELDS if field not in ("case_id", "seed")]
SUMMARY_FIELDS = GROUP_FIELDS + ["n_seeds", "mse_mean", "mse_std", "psnr_db_mean",
    "psnr_db_std", "psnr_is_infinite", "ssim_mean", "ssim_std", "runtime_ms", "runtime_iqr_ms", "measurement_run_id"]
TIMING_SCOPE = "algorithm_including_validation_and_conversions_excluding_io_noise_metrics"


def load_reference(image_path: Path, max_long_edge: int) -> tuple[np.ndarray, dict]:
    """Chuẩn hóa EXIF/RGB, chỉ thu nhỏ ảnh; kích thước metadata theo rộng × cao."""
    if type(max_long_edge) is not int or max_long_edge < 1:
        raise ValueError("Cạnh dài phải là số nguyên dương.")
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    source_bytes = image_path.read_bytes()
    try:
        with Image.open(image_path) as source:
            encoded_size = list(source.size)
            orientation = source.getexif().get(274, 1)
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            oriented_size = list(oriented.size)
            reference = np.array(oriented)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Không đọc được ảnh: {image_path}") from error

    height, width = reference.shape[:2]
    resize_scale = min(1.0, max_long_edge / max(height, width))
    if resize_scale < 1:
        working_size = (max(1, round(width * resize_scale)), max(1, round(height * resize_scale)))
        reference = cv2.resize(reference, working_size, interpolation=cv2.INTER_AREA)
    metadata = {
        "filename": image_path.name,
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "encoded_size": encoded_size,
        "exif_orientation": orientation,
        "oriented_size": oriented_size,
        "working_size": [reference.shape[1], reference.shape[0]],
        "color_space": "RGB",
        "dtype": "uint8",
        "data_range": 255,
        "resize_interpolation": "INTER_AREA",
    }
    return reference, metadata


def _require_rgb_uint8(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Ảnh phải là mảng RGB uint8 trong khoảng 0–255.")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("Ảnh phải có kích thước H×W×3 và không rỗng.")


def _require_kernel(kernel_size: int, allowed: tuple = (3, 5, 7, 9, 11)) -> None:
    if type(kernel_size) is not int or kernel_size not in allowed:
        raise ValueError(f"Kernel hỗ trợ: {allowed}.")


def _require_number(value: float, name: str, minimum: float = 0) -> None:
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} phải là số hữu hạn.")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} phải hữu hạn và không nhỏ hơn {minimum}.")


def make_noisy_image(
    reference: np.ndarray, noise_type: str, noise_level: float, seed: int
) -> tuple[np.ndarray, dict]:
    """Sinh nhiễu độc lập với thứ tự chạy filter; thống kê sau clip và làm tròn."""
    _require_rgb_uint8(reference)
    _require_number(noise_level, "Mức nhiễu")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed phải là số nguyên không âm.")
    if noise_type not in ("gaussian", "salt_pepper"):
        raise ValueError("Loại nhiễu phải là gaussian hoặc salt_pepper.")
    if noise_type == "salt_pepper" and noise_level > 1:
        raise ValueError("Salt-and-Pepper amount phải trong khoảng 0–1.")

    noise_code = 0 if noise_type == "gaussian" else 1
    # Mã mức nhiễu ổn định, không phụ thuộc hash ngẫu nhiên của tiến trình Python.
    level_digest = hashlib.sha256(float(noise_level).hex().encode()).digest()
    level_code = int.from_bytes(level_digest[:8], "big")
    entropy = [seed, noise_code, level_code]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    stats = {"noise_type": noise_type, "noise_level": noise_level, "seed": seed, "rng_entropy": entropy}
    if noise_type == "gaussian":
        noise = rng.normal(0, noise_level, reference.shape)
        result = np.rint(np.clip(reference.astype(np.float64) + noise, 0, 255)).astype(np.uint8)
        difference = result.astype(np.float64) - reference
        stats.update(mean=0, noise_sigma=noise_level,
            realized_mean=float(difference.mean()), realized_std=float(difference.std()))
    else:
        height, width = reference.shape[:2]
        selected_count = round(noise_level * height * width)
        indices = rng.choice(height * width, size=selected_count, replace=False)
        salt_count = round(selected_count / 2)
        result = reference.copy()
        flat = result.reshape(-1, 3)
        flat[indices[:salt_count]] = 255
        flat[indices[salt_count:]] = 0
        changed_count = int(np.count_nonzero(np.any(result != reference, axis=-1)))
        stats.update(amount=noise_level, salt_vs_pepper=0.5,
            selected_pixels=selected_count, changed_pixels=changed_count,
            selected_ratio=selected_count / (height * width),
            changed_ratio=changed_count / (height * width))
    return result, stats


def denoise(
    image: np.ndarray, method: str, kernel_size: int, filter_sigma: float | None = None
) -> np.ndarray:
    _require_rgb_uint8(image)
    _require_kernel(kernel_size)
    if method == "mean":
        return cv2.blur(image, (kernel_size, kernel_size), borderType=cv2.BORDER_REPLICATE)
    if method == "median":
        return cv2.medianBlur(image, kernel_size)
    if method == "gaussian":
        _require_number(filter_sigma, "Filter sigma")
        if filter_sigma == 0:
            raise ValueError("Filter sigma phải lớn hơn 0.")
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=filter_sigma,
            sigmaY=filter_sigma, borderType=cv2.BORDER_REPLICATE)
    raise ValueError("Phương pháp phải là mean, gaussian hoặc median.")


def sharpen_laplacian(image: np.ndarray, laplacian_kernel_size: int, alpha: float) -> np.ndarray:
    """Giữ dấu đạo hàm; chuẩn hóa gain khi thay aperture 3/5/7 của OpenCV."""
    _require_rgb_uint8(image)
    _require_kernel(laplacian_kernel_size, (3, 5, 7))
    _require_number(alpha, "Alpha")
    working = image.astype(np.float32)
    scale = 2.0 ** (4 - 2 * laplacian_kernel_size)
    laplacian = cv2.Laplacian(working, cv2.CV_32F, ksize=laplacian_kernel_size,
        scale=scale, borderType=cv2.BORDER_REPLICATE)
    return np.rint(np.clip(working - alpha * laplacian, 0, 255)).astype(np.uint8)


def compute_metrics(reference: np.ndarray, result: np.ndarray) -> dict:
    from skimage.metrics import structural_similarity

    _require_rgb_uint8(reference)
    _require_rgb_uint8(result)
    if reference.shape != result.shape or min(reference.shape[:2]) < 11:
        raise ValueError("Hai ảnh phải cùng kích thước và mỗi chiều ít nhất 11 pixel cho SSIM.")
    difference = reference.astype(np.float64) - result.astype(np.float64)
    mse = float(np.mean(difference ** 2))
    psnr = math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    ssim = structural_similarity(reference, result, data_range=255, channel_axis=-1,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return {"mse": mse, "psnr_db": psnr, "ssim": float(ssim)}


def _validate_config(config: dict) -> None:
    base_fields = set(DEFAULT_CONFIG) - {"selected_denoisers"}
    if not isinstance(config, dict) or set(config) not in (base_fields, base_fields | {"selected_denoisers"}):
        raise ValueError("Config phải chứa đúng các trường của DEFAULT_CONFIG.")
    if config["profile"] not in ("smoke", "full"):
        raise ValueError("Profile phải là smoke hoặc full.")
    for field in ("max_long_edge", "warmup", "repeats"):
        minimum = 11 if field == "max_long_edge" else (0 if field == "warmup" else 1)
        if type(config[field]) is not int or config[field] < minimum:
            raise ValueError(f"{field} phải là số nguyên không nhỏ hơn {minimum}.")
    for field in ("seeds", "save_seeds", "denoise_kernels", "gaussian_sigmas", "laplacian_kernels", "alphas"):
        values = config[field]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field} phải là danh sách không rỗng.")
        for value in values:
            if field in ("seeds", "save_seeds"):
                if type(value) is not int or value < 0:
                    raise ValueError("Seed phải là số nguyên không âm.")
            elif field == "laplacian_kernels":
                _require_kernel(value, (3, 5, 7))
            elif field == "denoise_kernels":
                _require_kernel(value)
            else:
                _require_number(value, field)
                if field == "gaussian_sigmas" and value == 0:
                    raise ValueError("Gaussian sigma phải lớn hơn 0.")
        if len(set(values)) != len(values):
            raise ValueError(f"{field} không được chứa giá trị trùng.")
    if not set(config["save_seeds"]).issubset(config["seeds"]):
        raise ValueError("save_seeds phải là tập con của seeds.")
    levels = config["noise_levels"]
    if not isinstance(levels, dict) or set(levels) != {"gaussian", "salt_pepper"}:
        raise ValueError("Config phải có cả Gaussian và Salt-and-Pepper.")
    for noise_type, values in levels.items():
        if not isinstance(values, list) or not values:
            raise ValueError("Mỗi loại nhiễu cần ít nhất một mức.")
        for value in values:
            _require_number(value, "Mức nhiễu")
            if noise_type == "salt_pepper" and value > 1:
                raise ValueError("Salt-and-Pepper amount phải trong khoảng 0–1.")
        if len(set(values)) != len(values):
            raise ValueError("Mức nhiễu không được trùng.")
    if "selected_denoisers" in config:
        settings = config["selected_denoisers"]
        if not isinstance(settings, dict) or set(settings) != set(levels):
            raise ValueError("Cấu hình denoiser phải có đủ hai loại nhiễu.")
        for noise_type, values in levels.items():
            if (not isinstance(settings[noise_type], dict)
                    or set(settings[noise_type]) != {str(float(v)) for v in values}):
                raise ValueError("Cấu hình denoiser phải khớp các mức nhiễu.")
            for value in values:
                methods = settings[noise_type][str(float(value))]
                if not isinstance(methods, dict) or set(methods) != {"mean", "gaussian", "median"}:
                    raise ValueError("Cần cấu hình cho cả Mean, Gaussian và Median.")
                for method, choice in methods.items():
                    if not isinstance(choice, list) or len(choice) != 2:
                        raise ValueError("Denoiser cần cặp [kernel, sigma].")
                    kernel, sigma = choice
                    _require_kernel(kernel)
                    if kernel not in config["denoise_kernels"]:
                        raise ValueError("Kernel đã chọn phải thuộc lưới thí nghiệm.")
                    if method == "gaussian":
                        _require_number(sigma, "Filter sigma")
                        if sigma <= 0 or sigma not in config["gaussian_sigmas"]:
                            raise ValueError("Sigma đã chọn phải thuộc lưới thí nghiệm.")
                    elif sigma is not None:
                        raise ValueError("Mean và Median không dùng sigma.")
    if not isinstance(config["roi_boxes"], dict) or not config["roi_boxes"]:
        raise ValueError("Cần ít nhất một ROI chuẩn hóa.")
    for name, box in config["roi_boxes"].items():
        if not isinstance(name, str) or not name.isidentifier() or len(box) != 4:
            raise ValueError("ROI cần tên hợp lệ và bốn tọa độ.")
        for value in box:
            _require_number(value, "Tọa độ ROI")
        x0, y0, x1, y1 = box
        if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
            raise ValueError("ROI chuẩn hóa phải nằm trong ảnh và có diện tích dương.")


def _make_case(base, stage, method, kernel=None, sigma=None, laplacian_kernel=None, alpha=None):
    case = dict(base, stage=stage, method=method, denoise_kernel_size=kernel,
        filter_sigma=sigma, laplacian_kernel_size=laplacian_kernel,
        laplacian_scale=None if laplacian_kernel is None else 2.0 ** (4 - 2 * laplacian_kernel), alpha=alpha)
    identity = json.dumps(case, sort_keys=True, allow_nan=False).encode()
    return dict(case_id=hashlib.sha256(identity).hexdigest()[:20], **case)


def selected_filter(noise_type: str, level: float, method: str, config: dict | None = None) -> tuple:
    config = DEFAULT_CONFIG if config is None else config
    if "selected_denoisers" in config:
        try:
            return tuple(config["selected_denoisers"][noise_type][str(float(level))][method])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Không có denoiser đã chọn cho loại/mức nhiễu này.") from error
    if method in ("mean", "median"):
        return 3, None
    if (method != "gaussian" or noise_type not in LEGACY_GAUSSIAN_SETTINGS
            or level not in LEGACY_GAUSSIAN_SETTINGS[noise_type]):
        raise ValueError("Cấu hình full chỉ dùng các loại/mức nhiễu đã chốt.")
    return LEGACY_GAUSSIAN_SETTINGS[noise_type][level]


def enumerate_cases(config: dict) -> list[dict]:
    """Liệt kê đầy đủ case; hai trục kernel luôn là các trường riêng biệt."""
    _validate_config(config)
    cases = []
    for noise_type in ("gaussian", "salt_pepper"):
        for level in config["noise_levels"][noise_type]:
            for seed in config["seeds"]:
                base = {"noise_type": noise_type, "noise_level": float(level), "seed": seed}
                cases.append(_make_case(base, "noisy", "none"))
                for method in ("mean", "gaussian", "median"):
                    for kernel in config["denoise_kernels"]:
                        sigmas = config["gaussian_sigmas"] if method == "gaussian" else [None]
                        for sigma in sigmas:
                            sigma = None if sigma is None else float(sigma)
                            cases.append(_make_case(base, "denoised", method, kernel, sigma))
                            if config["profile"] == "full":
                                selected_kernel, selected_sigma = selected_filter(noise_type, level, method, config)
                                if kernel != selected_kernel or sigma != selected_sigma:
                                    continue
                            for laplacian_kernel in config["laplacian_kernels"]:
                                for alpha in config["alphas"]:
                                    cases.append(_make_case(base, "denoised_then_sharpened", method,
                                        kernel, sigma, laplacian_kernel, float(alpha)))
                for laplacian_kernel in config["laplacian_kernels"]:
                    for alpha in config["alphas"]:
                        if config["profile"] == "full" and alpha not in (0.25, 0.5):
                            continue
                        cases.append(_make_case(base, "noisy_then_sharpened", "laplacian",
                            laplacian_kernel=laplacian_kernel, alpha=float(alpha)))
    return cases


def expected_counts(config: dict) -> dict:
    cases = enumerate_cases(config)
    processed = sum(case["stage"] != "noisy" for case in cases)
    saved = sum(case["seed"] in config["save_seeds"] for case in cases)
    return {"metrics": len(cases), "processed": processed,
        "timings": processed * config["repeats"], "images": saved + 1}


def check_environment() -> dict:
    """Dừng sớm nếu kernel không đáp ứng phiên bản/dependency của dự án."""
    if sys.version_info < (3, 13):
        raise RuntimeError("Part A cần Python >=3.13; hãy chọn kernel môi trường dự án.")
    packages = {}
    for name in ("numpy", "opencv-contrib-python-headless", "Pillow", "scikit-image", "matplotlib"):
        try:
            packages[name] = version(name)
        except Exception as error:
            raise RuntimeError(f"Thiếu dependency {name}; chạy uv sync --group dev.") from error
    return {"python": sys.version, "executable": sys.executable, "packages": packages}


def measure_runtime(operation: Callable[[], np.ndarray], warmup: int, repeats: int) -> list[float]:
    """Đo callable đã nhận ảnh trong RAM; không cộng các trung vị từng giai đoạn."""
    if type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("Warm-up phải >=0 và repeats phải >=1, đều là số nguyên.")
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return samples


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _save_image(path: Path, image: np.ndarray) -> str:
    Image.fromarray(image).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _apply_case(noisy: np.ndarray, case: dict) -> np.ndarray:
    stage = case["stage"]
    if stage == "noisy":
        return noisy.copy()
    result = noisy
    if stage in ("denoised", "denoised_then_sharpened"):
        result = denoise(result, case["method"], case["denoise_kernel_size"], case["filter_sigma"])
    if stage in ("denoised_then_sharpened", "noisy_then_sharpened"):
        result = sharpen_laplacian(result, case["laplacian_kernel_size"], case["alpha"])
    return result


def summarize_rows(rows: list[dict], timings: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    timing_groups = defaultdict(list)
    for timing in timings:
        timing_groups[timing["case_id"]].append(float(timing["elapsed_ms"]))
    for row in rows:
        if row["status"] == "ok":
            key = tuple(str(row.get(field, "")) for field in GROUP_FIELDS)
            groups[key].append(row)
    summaries = []
    for members in groups.values():
        summary = {field: members[0][field] for field in GROUP_FIELDS}
        summary["n_seeds"] = len(members)
        source_runs = {row.get("measurement_run_id", "") for row in members}
        if len(source_runs) != 1:
            raise ValueError("Không gộp timing của cùng cấu hình từ các run khác nhau.")
        summary["measurement_run_id"] = source_runs.pop()
        for metric in ("mse", "psnr_db", "ssim"):
            values = np.array([float(row[metric]) for row in members])
            summary[metric + "_mean"] = float(values.mean())
            summary[metric + "_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 and np.isfinite(values).all() else None
            )
        summary["psnr_is_infinite"] = bool(math.isinf(summary["psnr_db_mean"]))
        samples = [sample for row in members for sample in timing_groups[row["case_id"]]]
        summary["runtime_ms"] = float(np.median(samples)) if samples else None
        summary["runtime_iqr_ms"] = float(np.subtract(*np.percentile(samples, [75, 25]))) if samples else None
        summaries.append(summary)
    return summaries


def run_experiments(image_path: Path, output_root: Path, config: dict) -> Path:
    """Chạy một profile trong thư mục mới; lỗi xử lý được giữ lại trong CSV."""
    config = copy.deepcopy(config)
    cases = enumerate_cases(config)
    reference, metadata = load_reference(image_path, config["max_long_edge"])
    if min(reference.shape[:2]) < 11:
        raise ValueError("Ảnh tham chiếu quá nhỏ cho SSIM.")
    environment = check_environment()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = output_root / run_id
    run_dir.mkdir()
    (run_dir / "images").mkdir()
    manifest = {
        "run_id": run_id, "profile": config["profile"], "status": "running", "config": config,
        "input": metadata, "expected_counts": expected_counts(config), "noise_statistics": [],
        "environment": dict(environment, platform=platform.platform(),
            cpu=platform.processor() or platform.machine()),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "timing_scope": TIMING_SCOPE, "opencv_threads": 1, "opencl_enabled": False,
        "metrics": {"data_range": 255, "color_space": "RGB", "ssim_gaussian_weights": True,
            "ssim_sigma": 1.5, "ssim_use_sample_covariance": False},
    }
    manifest["source_code"] = Path(__file__).read_text()
    manifest["run_kind"] = "fresh_measurement"
    rows, timings = [], []
    previous_threads = cv2.getNumThreads()
    previous_opencl = cv2.ocl.useOpenCL()
    _write_json(run_dir / "run.json", manifest)
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        manifest["reference_sha256"] = _save_image(run_dir / "reference.png", reference)
        noisy_key, noisy = None, None
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as metrics_stream, \
             (run_dir / "timings.csv").open("w", newline="", encoding="utf-8") as timing_stream:
            metric_writer = csv.DictWriter(metrics_stream, fieldnames=METRIC_FIELDS)
            timing_writer = csv.DictWriter(timing_stream, fieldnames=TIMING_FIELDS)
            metric_writer.writeheader()
            timing_writer.writeheader()
            for case in cases:
                key = (case["noise_type"], case["noise_level"], case["seed"])
                row = dict.fromkeys(METRIC_FIELDS)
                row.update(case, status="ok", error="", runtime_reason="not_applicable",
                    image_saved="false", image_reason="seed_not_selected", measurement_run_id=run_id)
                case_timings = []
                try:
                    if key != noisy_key:
                        noisy, stats = make_noisy_image(reference, *key)
                        manifest["noise_statistics"].append(stats)
                        noisy_key = key
                    operation = lambda: _apply_case(noisy, case)
                    result = operation()
                    row.update(compute_metrics(reference, result))
                    row["psnr_is_infinite"] = str(math.isinf(row["psnr_db"])).lower()
                    if case["stage"] != "noisy":
                        samples = measure_runtime(operation, config["warmup"], config["repeats"])
                        row.update(runtime_ms=float(np.median(samples)),
                            runtime_iqr_ms=float(np.subtract(*np.percentile(samples, [75, 25]))),
                            runtime_reason="measured")
                        case_timings = [dict(case_id=case["case_id"], repeat=index,
                            elapsed_ms=sample, scope=TIMING_SCOPE) for index, sample in enumerate(samples)]
                    if case["seed"] in config["save_seeds"]:
                        relative = "images/" + case["case_id"] + ".png"
                        digest = _save_image(run_dir / relative, result)
                        row.update(image_saved="true", image_path=relative,
                            image_sha256=digest, image_reason="saved")
                except Exception as error:
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                rows.append(row)
                timings.extend(case_timings)
                metric_writer.writerow(row)
                timing_writer.writerows(case_timings)
                metrics_stream.flush()
                timing_stream.flush()
        _write_csv(run_dir / "summary.csv", summarize_rows(rows, timings), SUMMARY_FIELDS)
        _write_json(run_dir / "run.json", manifest)
        validation = validate_run(run_dir)
        _write_json(run_dir / "validation.json", validation)
        if not validation["valid"]:
            raise RuntimeError("Run không hợp lệ: " + "; ".join(validation["errors"][:5]))
        manifest["status"] = "complete"
        _write_json(run_dir / "run.json", manifest)
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json(run_dir / "run.json", manifest)
        raise
    finally:
        cv2.setNumThreads(previous_threads)
        cv2.ocl.setUseOpenCL(previous_opencl)
    return run_dir


def validate_run(run_dir: Path) -> dict:
    """Đối chiếu cấu hình, ảnh đầu ra và các bảng metric của run."""
    run_dir = Path(run_dir)
    errors = []
    try:
        manifest = json.loads((run_dir / "run.json").read_text())
        config = manifest["config"]
        expected = enumerate_cases(config)
        expected_by_id = {case["case_id"]: case for case in expected}
        rows = read_csv(run_dir / "metrics.csv")
        timings = read_csv(run_dir / "timings.csv")
        if manifest["status"] not in ("running", "complete"):
            errors.append("Run không có trạng thái running/complete.")
        if manifest["run_id"] != run_dir.name or manifest["profile"] != config["profile"]:
            errors.append("Run ID hoặc profile không khớp.")
        if len(rows) != len(expected) or {r["case_id"] for r in rows} != set(expected_by_id):
            errors.append("Danh sách case không khớp config.")
        if manifest["expected_counts"] != expected_counts(config):
            errors.append("Số lượng dự kiến không khớp cấu hình/case đã chọn.")
        if len(timings) != expected_counts(config)["timings"]:
            errors.append("Thiếu hoặc thừa mẫu timing.")
        reference_path = run_dir / "reference.png"
        if hashlib.sha256(reference_path.read_bytes()).hexdigest() != manifest["reference_sha256"]:
            errors.append("Checksum reference không khớp.")
        reference = np.array(Image.open(reference_path).convert("RGB"))
        timing_groups = defaultdict(list)
        for item in timings:
            elapsed = float(item["elapsed_ms"])
            if item["case_id"] not in expected_by_id or not math.isfinite(elapsed) or elapsed < 0:
                errors.append("Mẫu timing không hợp lệ.")
            if item["scope"] != TIMING_SCOPE:
                errors.append("Phạm vi timing không khớp.")
            timing_groups[item["case_id"]].append(item)
        for row in rows:
            case = expected_by_id[row["case_id"]]
            for field in CASE_FIELDS:
                wanted = "" if case[field] is None else str(case[field])
                if row[field] != wanted:
                    errors.append(f"Case {row['case_id']}: {field} không khớp.")
            if row["status"] != "ok":
                errors.append(f"Case lỗi {row['case_id']}: {row['error']}")
                continue
            mse, psnr, ssim = (float(row[key]) for key in ("mse", "psnr_db", "ssim"))
            if not math.isfinite(mse) or mse < 0 or not math.isfinite(ssim) or not -1 <= ssim <= 1:
                errors.append("Metric ngoài miền hợp lệ.")
            expected_psnr = math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse) if mse > 0 else math.nan
            if not math.isclose(psnr, expected_psnr, rel_tol=1e-9, abs_tol=1e-9):
                errors.append("PSNR không khớp MSE.")
            samples = timing_groups[row["case_id"]]
            if case["stage"] == "noisy":
                if samples or row["runtime_ms"] or row["runtime_reason"] != "not_applicable":
                    errors.append("Baseline noisy không được có runtime filter.")
            else:
                if (len(samples) != config["repeats"]
                        or {int(t["repeat"]) for t in samples} != set(range(config["repeats"]))):
                    errors.append("Số mẫu timing của case không đúng.")
                if samples and not math.isclose(
                        float(row["runtime_ms"]),
                        float(np.median([float(t["elapsed_ms"]) for t in samples]))):
                    errors.append("Runtime median không khớp mẫu gốc.")
            should_save = case["seed"] in config["save_seeds"]
            if (row["image_saved"] == "true") != should_save:
                errors.append("Trạng thái lưu ảnh không khớp seed.")
            if should_save:
                wanted_path = "images/" + row["case_id"] + ".png"
                if row["image_path"] != wanted_path:
                    errors.append("Đường dẫn ảnh không khớp case.")
                    continue
                path = run_dir / wanted_path
                if hashlib.sha256(path.read_bytes()).hexdigest() != row["image_sha256"]:
                    errors.append("Checksum ảnh không khớp.")
                result = np.array(Image.open(path).convert("RGB"))
                recalculated = compute_metrics(reference, result)
                for metric in ("mse", "psnr_db", "ssim"):
                    if not math.isclose(float(row[metric]), recalculated[metric], rel_tol=1e-9, abs_tol=1e-9):
                        errors.append(f"{metric} không khớp PNG.")
        summaries = read_csv(run_dir / "summary.csv")
        recalculated_summary = summarize_rows(rows, timings)
        if len(summaries) != len(recalculated_summary):
            errors.append("Số dòng summary không khớp.")
        else:
            for actual, wanted in zip(summaries, recalculated_summary, strict=True):
                for field in SUMMARY_FIELDS:
                    value = wanted[field]
                    if value is None:
                        equal = actual[field] == ""
                    elif isinstance(value, float):
                        equal = math.isclose(float(actual[field]), value, rel_tol=1e-9, abs_tol=1e-9)
                    else:
                        equal = actual[field] == str(value)
                    if not equal:
                        errors.append(f"Summary {field} không khớp CSV gốc.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"Không đọc/kiểm tra được artifact: {error}")
    return {"valid": not errors, "errors": errors}


def choose_denoiser(rows: list[dict], method: str, config: dict | None) -> dict:
    candidates = [row for row in rows if row["method"] == method and row["stage"] == "denoised"]
    if config is None or config["profile"] != "full":
        return select_best_denoised(candidates)
    kernel, sigma = selected_filter(candidates[0]["noise_type"], float(candidates[0]["noise_level"]), method, config)
    return next(row for row in candidates if int(row["denoise_kernel_size"]) == kernel
        and (float(row["filter_sigma"]) if row["filter_sigma"] else None) == sigma)


def resolve_rois(normalized_boxes: dict, image_shape: tuple) -> dict:
    height, width = image_shape[:2]
    boxes = {}
    for name, coordinates in normalized_boxes.items():
        if len(coordinates) != 4:
            raise ValueError("ROI cần bốn tọa độ.")
        x0, y0, x1, y1 = coordinates
        if not all(math.isfinite(v) for v in coordinates) or not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("ROI chuẩn hóa nằm ngoài ảnh.")
        box = (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("ROI quá nhỏ sau khi đổi tọa độ.")
        boxes[name] = box
    return boxes

def select_best_denoised(rows: list[dict]) -> dict:
    candidates = [row for row in rows if row["stage"] == "denoised"]
    if not candidates:
        raise ValueError("Không có cấu hình denoised.")
    return min(candidates, key=lambda row: (-float(row["ssim_mean"]),
        -float(row["psnr_db_mean"]), json.dumps(row, sort_keys=True)))

def comparison_tables(summary_rows: list[dict], config: dict | None = None) -> dict[str, list[dict]]:
    """Ghép chuỗi denoise–sharpen theo cấu hình chốt; smoke chọn theo SSIM."""
    variants = sorted({(row["noise_type"], row["noise_level"]) for row in summary_rows})
    best_rows, sharpened_rows = [], []
    for noise_type, level in variants:
        members = [row for row in summary_rows
            if row["noise_type"] == noise_type and row["noise_level"] == level]
        noisy = next(row for row in members if row["stage"] == "noisy")
        methods = sorted({row["method"] for row in members if row["stage"] == "denoised"})
        for method in methods:
            best = choose_denoiser(members, method, config)
            best_rows.append(dict(best,
                delta_ssim_vs_noisy=float(best["ssim_mean"]) - float(noisy["ssim_mean"]),
                delta_psnr_db_vs_noisy=float(best["psnr_db_mean"]) - float(noisy["psnr_db_mean"])))
            matching = [row for row in members if row["stage"] == "denoised_then_sharpened"
                and row["method"] == method
                and row["denoise_kernel_size"] == best["denoise_kernel_size"]
                and row["filter_sigma"] == best["filter_sigma"]]
            for row in matching:
                sharpened_rows.append(dict(row,
                    denoised_ssim_mean=best["ssim_mean"], denoised_psnr_db_mean=best["psnr_db_mean"],
                    denoised_runtime_ms=best["runtime_ms"],
                    delta_ssim_vs_denoised=float(row["ssim_mean"]) - float(best["ssim_mean"]),
                    delta_psnr_db_vs_denoised=float(row["psnr_db_mean"]) - float(best["psnr_db_mean"])))
    return {
        "denoised_comparison.csv": best_rows,
        "sharpening_comparison.csv": sharpened_rows,
        "direct_sharpening.csv": [row for row in summary_rows if row["stage"] == "noisy_then_sharpened"],
        "noisy_baselines.csv": [row for row in summary_rows if row["stage"] == "noisy"],
    }

def _label(row: dict) -> str:
    if row["stage"] == "noisy":
        return f"Noisy: {row['noise_type']}\nlevel={row['noise_level']}, seed={row['seed']}"
    if row["method"] == "laplacian":
        return f"Laplacian k={row['laplacian_kernel_size']}, alpha={row['alpha']}"
    label = row["method"].title()
    if row.get("denoise_kernel_size"):
        label += " k=" + row["denoise_kernel_size"]
    if row.get("filter_sigma"):
        label += " sigma=" + row["filter_sigma"]
    if row.get("laplacian_kernel_size"):
        label += f"\nLaplacian k={row['laplacian_kernel_size']}, alpha={row['alpha']}"
    return label

def build_figures(run_dir: Path, roi_boxes: dict[str, tuple[int, int, int, int]]) -> list[Path]:
    import textwrap

    run_dir = Path(run_dir)
    validation = validate_run(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text())
    if not validation["valid"] or manifest["status"] != "complete":
        raise ValueError("Chỉ tạo hình từ run complete và artifact hợp lệ.")
    _write_json(run_dir / "validation.json", validation)
    reference = np.array(Image.open(run_dir / "reference.png").convert("RGB"))
    height, width = reference.shape[:2]
    for name, box in roi_boxes.items():
        if not name.isidentifier() or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("ROI cần tên hợp lệ và tọa độ nguyên.")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("ROI nằm ngoài ảnh.")
    if not roi_boxes:
        raise ValueError("Cần ít nhất một ROI.")
    figure_dir = run_dir / "figures"
    table_dir = run_dir / "tables"
    figure_dir.mkdir(exist_ok=True)
    table_dir.mkdir(exist_ok=True)

    rows = read_csv(run_dir / "metrics.csv")
    summary = read_csv(run_dir / "summary.csv")
    config = manifest["config"]
    paths = [run_dir / "summary.csv"]
    for filename, table_rows in comparison_tables(summary, config).items():
        if not table_rows:
            continue
        path = table_dir / filename
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
            writer.writeheader()
            writer.writerows(table_rows)
        paths.append(path)
    records = []
    title = f"{manifest['profile']} | {manifest['run_id']}"

    def save(fig, name, cases=(), note="", h_pad=2.5):
        fig.suptitle(title + ("\n" + note if note else ""), fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=h_pad, w_pad=1.0)
        for suffix in ("png", "pdf"):
            path = figure_dir / f"{name}.{suffix}"
            fig.savefig(path, dpi=300)
            paths.append(path)
        plt.close(fig)
        records.append({"name": name, "case_ids": [row["case_id"] for row in cases], "note": note})

    def pixels(row):
        return np.array(Image.open(run_dir / row["image_path"]).convert("RGB"))

    def panel_label(row):
        return "\n".join(textwrap.fill(line, width=20) for line in _label(row).splitlines())

    roi_rows = []
    for row in rows:
        if row["image_saved"] != "true":
            continue
        output = pixels(row)
        for name, (x0, y0, x1, y1) in roi_boxes.items():
            record = {"case_id": row["case_id"], "seed": row["seed"], "roi": name,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "status": "ok", "mse": "", "psnr_db": "", "ssim": ""}
            if min(x1-x0, y1-y0) < 11:
                record["status"] = "too_small_for_ssim"
            else:
                measured = compute_metrics(reference[y0:y1, x0:x1], output[y0:y1, x0:x1])
                record.update({key: measured[key] for key in ("mse", "psnr_db", "ssim")})
            roi_rows.append(record)
    if roi_rows:
        path = table_dir / "roi_metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(roi_rows[0]))
            writer.writeheader()
            writer.writerows(roi_rows)
        paths.append(path)

    def contributing_cases(summary_members):
        keys = {tuple(member[field] for field in GROUP_FIELDS) for member in summary_members}
        return [row for row in rows if tuple(row[field] for field in GROUP_FIELDS) in keys]

    def comparison_panel(selected, name, crop=None):
        columns = min(3, len(selected) + 1)
        row_count = math.ceil((len(selected) + 1) / columns)
        fig, axes = plt.subplots(row_count, columns, figsize=(6.4, 3 * row_count), squeeze=False)
        arrays = [reference] + [pixels(row) for row in selected]
        labels = ["Clean reference"] + [panel_label(row) for row in selected]
        for axis in axes.flat:
            axis.axis("off")
        for axis, array, label in zip(axes.flat[:len(arrays)], arrays, labels, strict=True):
            if crop is not None:
                x0, y0, x1, y1 = crop
                array = array[y0:y1, x0:x1]
            axis.imshow(array, interpolation="nearest")
            axis.set_title(label, fontsize=8)
            axis.axis("off")
        save(fig, name, selected, "Fixed input, same ROI and nearest-neighbor display")

    def panel_grid(groups, name, crop=None):
        fig, axes = plt.subplots(len(groups), 4, figsize=(6.4, 2.5 * len(groups)), squeeze=False)
        selected = []
        for axes_row, (noisy, denoised, sharpened) in zip(axes, groups, strict=True):
            selected.extend((noisy, denoised, sharpened))
            arrays = [reference, pixels(noisy), pixels(denoised), pixels(sharpened)]
            labels = ["Clean reference", panel_label(noisy), panel_label(denoised), panel_label(sharpened)]
            for axis, array, label in zip(axes_row, arrays, labels, strict=True):
                if crop is not None:
                    x0, y0, x1, y1 = crop
                    array = array[y0:y1, x0:x1]
                axis.imshow(array, interpolation="nearest")
                axis.set_title(label, fontsize=8)
                axis.axis("off")
        noisy = groups[0][0]
        save(fig, name, selected,
            f"{noisy['noise_type']}, level={noisy['noise_level']}, seed={noisy['seed']}; "
            "same reference and noisy input")

    fig, axis = plt.subplots(figsize=(6, 8))
    axis.imshow(reference)
    for name, (x0, y0, x1, y1) in roi_boxes.items():
        axis.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="red"))
        axis.text(x0, y0, name, color="black", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
    axis.axis("off")
    save(fig, "reference_rois")

    for noise_type in ("gaussian", "salt_pepper"):
        levels = config["noise_levels"][noise_type]
        level = levels[len(levels)//2]
        seed = config["save_seeds"][0]
        available = [row for row in rows if row["noise_type"] == noise_type
            and float(row["noise_level"]) == level and int(row["seed"]) == seed]
        noisy = next(row for row in available if row["stage"] == "noisy")
        groups = []
        for method in ("mean", "gaussian", "median"):
            candidates = [row for row in summary if row["noise_type"] == noise_type
                and float(row["noise_level"]) == level and row["method"] == method]
            best = choose_denoiser(candidates, method, config)
            denoised = next(row for row in available if row["stage"] == "denoised"
                and row["method"] == method and row["denoise_kernel_size"] == best["denoise_kernel_size"]
                and row["filter_sigma"] == best["filter_sigma"])
            sharpened = next(row for row in available if row["stage"] == "denoised_then_sharpened"
                and row["method"] == method and row["denoise_kernel_size"] == denoised["denoise_kernel_size"]
                and row["filter_sigma"] == denoised["filter_sigma"]
                and int(row["laplacian_kernel_size"]) == config["laplacian_kernels"][0]
                and float(row["alpha"]) == config["alphas"][0])
            groups.append((noisy, denoised, sharpened))
        panel_grid(groups, noise_type + "_grid")
        for name, box in roi_boxes.items():
            panel_grid(groups, noise_type + "_roi_" + name, box)

        if len(config["denoise_kernels"]) > 1:
            kernel_rows = [row for row in available if row["stage"] == "denoised"
                and row["method"] == "mean" and int(row["denoise_kernel_size"]) in
                (min(config["denoise_kernels"]), max(config["denoise_kernels"]))]
            for name, box in roi_boxes.items():
                comparison_panel(kernel_rows, noise_type + "_kernel_comparison_" + name, box)
        if len(config["laplacian_kernels"]) > 1 or len(config["alphas"]) > 1:
            base = groups[0][1]
            sharpen_rows = [row for row in available if row["stage"] == "denoised_then_sharpened"
                and row["method"] == base["method"]
                and row["denoise_kernel_size"] == base["denoise_kernel_size"]
                and row["filter_sigma"] == base["filter_sigma"]
                and int(row["laplacian_kernel_size"]) == config["laplacian_kernels"][0]]
            for name, box in roi_boxes.items():
                comparison_panel([base] + sharpen_rows,
                    noise_type + "_sharpening_comparison_" + name, box)

        comparisons = [triple[1] for triple in groups] + [groups[0][2]]
        errors = [np.abs(pixels(row).astype(float) - reference).mean(axis=-1) for row in comparisons]
        error_max = max(1.0, max(float(error.max()) for error in errors))
        fig, axes = plt.subplots(1, len(errors), figsize=(12, 4))
        for axis, error, row in zip(axes, errors, comparisons, strict=True):
            display = axis.imshow(error, cmap="magma", vmin=0, vmax=error_max)
            axis.set_title(_label(row), fontsize=8)
            axis.axis("off")
        # Colorbar chung cần bố cục thủ công, không ép tight_layout.
        fig.suptitle(title + "\nCommon error scale", fontsize=10)
        fig.subplots_adjust(top=0.72, right=0.82)
        colorbar_axis = fig.add_axes((0.86, 0.22, 0.012, 0.40))
        fig.colorbar(display, cax=colorbar_axis, label="Mean absolute RGB error (0–255)")
        name = noise_type + "_error_map"
        for suffix in ("png", "pdf"):
            path = figure_dir / f"{name}.{suffix}"
            fig.savefig(path, dpi=300)
            paths.append(path)
        plt.close(fig)
        records.append({"name": name, "case_ids": [row["case_id"] for row in comparisons], "note": "common scale"})

        fig, axes = plt.subplots(3, 1, figsize=(6.4, 8.4))
        for method in ("mean", "gaussian", "median"):
            members = [row for row in summary if row["noise_type"] == noise_type
                and float(row["noise_level"]) == level and row["stage"] == "denoised"
                and row["method"] == method]
            sigmas = sorted({row["filter_sigma"] for row in members})
            for sigma in sigmas:
                curve = sorted([row for row in members if row["filter_sigma"] == sigma],
                    key=lambda row: int(row["denoise_kernel_size"]))
                label = method + (" sigma=" + sigma if sigma else "")
                kernels = [int(row["denoise_kernel_size"]) for row in curve]
                axes[0].plot(kernels, [float(row["psnr_db_mean"]) for row in curve], "o-", label=label)
                axes[1].plot(kernels, [float(row["ssim_mean"]) for row in curve], "o-", label=label)
                axes[2].scatter([float(row["runtime_ms"]) for row in curve],
                    [float(row["ssim_mean"]) for row in curve], label=label)
        for axis, xlabel, ylabel in zip(axes, ["Denoise kernel", "Denoise kernel", "Runtime median (ms)"],
                ["PSNR mean (dB)", "SSIM mean", "SSIM mean"], strict=True):
            axis.set(xlabel=xlabel, ylabel=ylabel)
            axis.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        note = ("Single setting: no sensitivity conclusion"
                if len(config["denoise_kernels"]) == 1
                else "Quality averaged over seeds; timing pooled median")
        sensitivity_members = [row for row in summary if row["noise_type"] == noise_type
            and float(row["noise_level"]) == level and row["stage"] == "denoised"]
        save(fig, noise_type + "_sensitivity", contributing_cases(sensitivity_members), note)

        gaussian = [row for row in summary if row["noise_type"] == noise_type
            and float(row["noise_level"]) == level and row["stage"] == "denoised" and row["method"] == "gaussian"]
        heat = np.array([[float(next(row for row in gaussian if int(row["denoise_kernel_size"]) == kernel
            and float(row["filter_sigma"]) == sigma)["ssim_mean"]) for sigma in config["gaussian_sigmas"]]
            for kernel in config["denoise_kernels"]])
        fig, axis = plt.subplots(figsize=(5, 4))
        display = axis.imshow(heat, aspect="auto")
        axis.set_xticks(range(len(config["gaussian_sigmas"])), config["gaussian_sigmas"])
        axis.set_yticks(range(len(config["denoise_kernels"])), config["denoise_kernels"])
        axis.set(xlabel="Filter sigma", ylabel="Denoise kernel")
        fig.colorbar(display, ax=axis, label="SSIM mean")
        save(fig, noise_type + "_gaussian_heatmap", contributing_cases(gaussian))

        sharp_rows = [row for row in available if row["stage"] == "noisy_then_sharpened"]
        columns = min(4, len(sharp_rows) + 1)
        row_count = math.ceil((len(sharp_rows) + 1) / columns)
        fig, axes = plt.subplots(row_count, columns, figsize=(6.4, 3.5 * row_count), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, row in zip(axes.flat[:len(sharp_rows)+1], [noisy] + sharp_rows, strict=True):
            axis.imshow(pixels(row), interpolation="nearest")
            axis.set_title(panel_label(row), fontsize=8)
            axis.axis("off")
        save(fig, noise_type + "_sharpening_direct", [noisy] + sharp_rows,
            "Direct sharpening is not denoising", h_pad=7)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sharpening_members = []
        for method in ("mean", "gaussian", "median"):
            for laplacian_kernel in config["laplacian_kernels"]:
                curve = [row for row in summary if row["noise_type"] == noise_type
                    and float(row["noise_level"]) == level and row["stage"] == "denoised_then_sharpened"
                    and row["method"] == method
                    and row["denoise_kernel_size"] == next(
                        group[1]["denoise_kernel_size"] for group in groups
                        if group[1]["method"] == method)
                    and row["filter_sigma"] == next(
                        group[1]["filter_sigma"] for group in groups
                        if group[1]["method"] == method)
                    and int(row["laplacian_kernel_size"]) == laplacian_kernel]
                curve.sort(key=lambda row: float(row["alpha"]))
                sharpening_members.extend(curve)
                for axis, metric in zip(axes, ("psnr_db_mean", "ssim_mean"), strict=True):
                    axis.plot([float(row["alpha"]) for row in curve], [float(row[metric]) for row in curve],
                        "o-", label=f"{method}, L={laplacian_kernel}")
        for axis, ylabel in zip(axes, ("PSNR mean (dB)", "SSIM mean"), strict=True):
            axis.set(xlabel="Sharpening alpha", ylabel=ylabel)
            axis.legend(fontsize=6)
        save(fig, noise_type + "_sharpening_sensitivity", contributing_cases(sharpening_members),
            "Selected denoiser per method; fixed kernel/sigma across alpha")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        noise_level_members = []
        for method in ("mean", "gaussian", "median"):
            curve = []
            for noise_level in levels:
                candidates = [row for row in summary if row["noise_type"] == noise_type
                    and float(row["noise_level"]) == noise_level and row["method"] == method]
                best = choose_denoiser(candidates, method, config)
                curve.append((noise_level, best))
                noise_level_members.append(best)
            for axis, metric in zip(axes, ("psnr_db_mean", "ssim_mean"), strict=True):
                axis.plot([item[0] for item in curve], [float(item[1][metric]) for item in curve], "o-", label=method)
        for axis, ylabel in zip(axes, ("PSNR mean (dB)", "SSIM mean"), strict=True):
            axis.set(xlabel="Noise sigma" if noise_type == "gaussian" else "Impulse amount", ylabel=ylabel)
            axis.legend(fontsize=7)
        save(fig, noise_type + "_noise_levels", contributing_cases(noise_level_members),
            "Fixed selected denoiser per level")

    payload = {"run_id": manifest["run_id"], "profile": manifest["profile"], "roi_boxes": roi_boxes,
        "figures": records, "source": "metrics.csv and summary.csv from this run",
        "figure_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    mapping_path = figure_dir / "figures.json"
    mapping_path.write_text(json.dumps(payload, indent=2) + "\n")
    paths.append(mapping_path)
    return paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Part A: filtering and enhancement")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    config = SMOKE_CONFIG if args.profile == "smoke" else DEFAULT_CONFIG
    run_dir = run_experiments(args.input, args.output, config)
    reference, _ = load_reference(args.input, config["max_long_edge"])
    build_figures(run_dir, resolve_rois(config["roi_boxes"], reference.shape))
    print(run_dir.resolve())


if __name__ == "__main__":
    main()
