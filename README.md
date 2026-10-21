# INS3155 — Computer Vision Midterm

**Traditional Image Processing for Filtering, 3D Reconstruction, and Image Stitching**

Python implementations and executed notebooks for three classical computer vision tasks:

- **A:** Gaussian/Salt-and-Pepper corruption, Mean/Gaussian/Median denoising and Laplacian sharpening.
- **B:** Venus stereo correspondences, RANSAC fundamental matrices, BM/SGBM disparity and coloured point clouds.
- **C:** Four-image stitching with SIFT, ORB and AKAZE, explicit homographies, feather and multi-band blending.

The English report source is [report_source/main.tex](report_source/main.tex). It covers all three parts, with methodology, measured results, comparative analysis, references, Python excerpts and a complete final-grid Part A appendix. Build it as described in [Report](#8-report). The compiled PDF is written to `report_source/build/main.pdf`.

## 1. Project files

| Path | Contents |
| --- | --- |
| [src/part_a.py](src/part_a.py) | Filtering, corruption, metrics, timing and result export |
| [src/part_b.py](src/part_b.py) | Stereo geometry, dense matching, masks, camera model and PLY export |
| [src/part_c.py](src/part_c.py) | Features, matching, RANSAC H, warping, blending and comparisons |
| [notebooks/part_a.ipynb](notebooks/part_a.ipynb), [part_b.ipynb](notebooks/part_b.ipynb), [part_c.ipynb](notebooks/part_c.ipynb) | Parameters and runnable workflows; generated outputs are stored under `results/` |
| `data/part_a/`, `data/part_b/Venus/`, `data/part_c/` | Local inputs and metadata; supplied separately from Git |
| `results/part_a/`, `results/part_b/`, `results/part_c/` | Run manifests, CSV/NPZ measurements, figures and reconstructed outputs |
| [report_source/main.tex](report_source/main.tex) | English submission report covering A/B/C |
| [report_source/part_a_full_results.tex](report_source/part_a_full_results.tex) | Complete 528-group Part A results appendix included by `main.tex` |
| [pyproject.toml](pyproject.toml), [uv.lock](uv.lock) | Dependency declarations and resolved versions |

## 2. Installation

Use Python **3.13 or later** and [uv](https://docs.astral.sh/uv/). Run commands from the project root:

```bash
uv sync --locked --group dev
```

The locked environment supplies NumPy, Matplotlib, Pillow, scikit-image and `opencv-contrib-python-headless==5.0.0.93`. The contrib package provides `cv2.xfeatures2d.AKAZE_create` used by OpenCV 5. Keep one installed distribution supplying `cv2`. The development group includes JupyterLab, ipykernel and notebook execution tools.

Recorded results use Python 3.13.14, NumPy 2.5.4, Matplotlib 3.11.2 and OpenCV 5.0.0. Part A also records Pillow 13.0.0 and scikit-image 0.26.0. Its producer used `opencv-python-headless==5.0.0.93`; B/C used the contrib distribution at the same version. Each run records its environment rather than assigning the current environment to old measurements.

## 3. Input data

Restore the supplied local data bundle before running A/C. Download the Venus files from the official archives for B. Code-only checkouts omit input images and metadata. Recorded result directories are also supplied separately when needed to display the report's linked figures.

```text
data/
├── part_a/
│   ├── succulent_20231108_171619.jpg
│   └── metadata.json
├── part_b/Venus/
│   ├── frame10.png
│   ├── frame11.png
│   ├── flow10.flo
│   └── metadata.json
└── part_c/
    ├── pagoda_01.png
    ├── pagoda_02.png
    ├── pagoda_03.png
    ├── pagoda_04.png
    └── metadata.json
```

### Part A: reference photograph

The supplied succulent JPEG contains leaf edges, gravel texture and a glazed pot. Its encoded size is 4128×3096 with EXIF orientation 6. The loader corrects orientation, converts to RGB and area-resizes to **768×1024**, without cropping. Fixed pot/leaf/gravel crops supplement full-image scores. The source JPEG is preserved. Its photographer and acquisition source have not been confirmed in the available metadata.

SHA-256: `ee48060d10e9c8e2ed6829860c4ef25144928029b6ebb755deb1c2ae449e8937`.

### Part B: Middlebury Flow Venus

Use **Venus** from [Middlebury Optical Flow — Other Datasets](https://vision.middlebury.edu/flow/data/):

- [Colour two-frame archive](https://vision.middlebury.edu/flow/data/comp/zip/other-color-twoframes.zip): extract Venus `frame10.png` and `frame11.png`.
- [Ground-truth flow archive](https://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip): extract Venus `flow10.flo`.

Place these files together with the supplied metadata in `data/part_b/Venus/`. Both images are **420×380**; frame 10 is left/reference and frame 11 right/target. No further resize, crop or rectification is applied. The metadata records checksums, archive URLs, zero vertical GT flow and crop-coordinate verification. The archive links and per-file checksums identify the Middlebury inputs; A/C use the supplied local files.

| File | SHA-256 |
| --- | --- |
| `frame10.png` | `f36d77900848c4d5bf6f31898e43d3db4c912f474f273b331d7a96113ea5a637` |
| `frame11.png` | `45860656c48fec7fff1ed3528bd2d72f4f6dbf71c6d621fb2e0f48a531e617d0` |
| `flow10.flo` | `4f5e58609d02d8198f838de8b3f34a952cfaebf284938daa255066c535610f34` |

### Part C: four overlapping views

The student photographed Chua Minh Thanh during a trip to Gia Lai. Use the supplied four-image interior sequence in the listed order; this source information is recorded in `data/part_c/metadata.json`. All four PNGs are **1920×1080**. Image 3 anchors direct transformations **1→3, 2→3, 4→3**. No input resizing or cropping is performed. The loader validates file count, decoded dimensions and metadata checksums.

| File | SHA-256 |
| --- | --- |
| `pagoda_01.png` | `735623ea8fae4450288e2b5268db7b3124a6089f717377277fcd516b8b0c3e4a` |
| `pagoda_02.png` | `04905cc3accb16bceaa6a1ea0e9c808c611b302321e99935783917f862bed569` |
| `pagoda_03.png` | `60ab5272174ec3d509e2ad9f25059187f5b5634febef1273cb2c452abb41705a` |
| `pagoda_04.png` | `9b79b8a48ee0ad43f888cba15f87ad1d26a1c117776dcfae17ea232a56a50ced` |

## 4. Run Part A

```bash
uv run python src/part_a.py --profile full
uv run jupyter lab notebooks/part_a.ipynb
```

The CLI creates a new timestamp/UUID run under `results/part_a/official/`, exports the full parameter comparison and figures, and preserves existing runs. `--input` and `--output` override paths. The notebook reads the recorded official run when present, checks measurement-source compatibility and validation, and adds `tables/recommended_outputs.csv` and `tables/objective_roi_comparison.csv`. Run all notebook cells to reproduce these additional report tables. Set `rerun_pipeline = True` in its run cell for a fresh measurement; an absent recorded directory also triggers a new run.

| Parameter | Full setting |
| --- | --- |
| Gaussian noise σₙ | 10, 20, 30 intensity units |
| Salt-and-Pepper density | 1%, 3%, 5%; distinct spatial pixels |
| Seeds | 42, 43, 44; seed 42 saved as images |
| Denoiser kernels | 3, 5, 7, 9, 11 |
| Gaussian filter σ_f | 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6 |
| Laplacian aperture / scale | 3/¼, 5/¹⁄₆₄, 7/¹⁄₁₀₂₄; scale `2**(4-2*k)` |
| Post-denoising α | 0.1, 0.2, 0.25, 0.5 |
| Timing | 2 warmups + 10 repetitions per seed; median (IQR) over 30 samples |

Quality is mean ± sample SD over three seeds. MSE uses float64 RGB differences, PSNR uses peak 255, and SSIM uses explicit `data_range=255`, channel averaging, Gaussian weights (sigma 1.5) and population covariance. Timing includes algorithm checks/conversions and excludes noise generation, metrics, I/O and plotting. Complete denoise–sharpen chains are timed together.

The report uses the fixed official grid above. Its selected complete chains use Gaussian 5×5/σ0.8, 7×7/σ1.2 and 9×9/σ1.5 with Laplacian 7/α0.1 for Gaussian σₙ10/20/30, respectively. For impulse density 1/3/5%, they use Median 3×3 and Laplacian 5 with α0.25/0.25/0.2.

## 5. Run Part B

```bash
uv run python src/part_b.py --profile full
uv run jupyter lab notebooks/part_b.ipynb
```

`--input-dir` and `--output` override the default Venus input and `results/part_b/` output. The notebook finds a compatible saved run. Set `rerun_pipeline = True` to measure a new run, or set `run_directory` to inspect a particular saved run.

| Parameter | Full setting |
| --- | --- |
| Sparse matching | SIFT; BF L2 KNN k=2; ratio 0.75 |
| F estimation | USAC_MAGSAC; thresholds 0.5/1/2 px; confidence 0.999; max 5,000 iterations |
| Dense methods | BM and SGBM 3-way |
| Disparity grid | `minDisparity=-16`; `numDisparities=32,64`; `blockSize=5,9,15` |
| Validity | Native sentinel/range checks, in-bounds target, LR tolerance 1 px |
| Timing | 3 warmups + 20 repetitions; median, min/max/std/CV retained |
| Cloud reference settings | BM/SGBM, 32 disparities, block 9 |

For this verified horizontal pair, `d_gt=-flow_u`, including zero/negative disparities. Raw/LR valid ratios divide by all 159,600 image pixels. RMSE and Bad >1 px use the intersection with the 157,906-pixel GT-valid/in-bounds domain. Coverage uses that fixed GT domain. Bad-or-invalid also counts rejected predictions. `common_domain_metrics.csv` compares BM/SGBM on identical accepted pixels.

The camera inherits the 30° viewing-angle choice from **Basha, Moses and Kiryati (2013), §4.1**, [DOI 10.1007/s11263-012-0542-7](https://doi.org/10.1007/s11263-012-0542-7). The project interprets it as horizontal FoV and selects baseline B=1, parallel axes, square pixels, zero skew and centered source principal points. Unequal source crops give `cx_left=206.5`, `cx_right=216.5`, `cy=191`, `f=809.855025 px`. Reprojection uses `Z=f/(d+10)` and produces **normalized baseline units**. Disparity metrics keep the cropped-image convention `d=-u`. The derivation and assumptions are documented in the Part B methodology of [main.tex](report_source/main.tex).

Point clouds retain finite positive-depth LR-consistent samples. `cloud.ply` preserves raw points; `cloud_filtered.ply` removes small prediction-connected disparity components. Their masks, camera model and depth-display bounds are saved with the results.

## 6. Run Part C

### Coarse grid

```bash
uv run python src/part_c.py --profile full
```

This runs **27 coarse configurations**: SIFT/ORB/AKAZE × ratios 0.70/0.75/0.80 × thresholds 2/3/5 px. It exports panoramas, feature/match figures and geometric records under `results/part_c/`, leaving selection pending. Use `--input-dir` and `--output` to override paths.

### Refined comparison used in the report

```bash
uv run jupyter lab notebooks/part_c.ipynb
```

Run cells in order with the environment's Python kernel. The notebook reads the saved refinement run with **36 configurations**: three detectors × ratios 0.72/0.75/0.78 × thresholds 3/4/5/6 px. `run_directory` can select a particular saved run.

To reproduce the report's refinement experiment, set `rerun_pipeline = True` in the notebook and run all cells in order. This runs the 36-configuration grid, applies the recorded selections, repeats timing, analyzes sensitivity and compares blending. The selected settings are **SIFT 0.75/5 px, ORB 0.78/5 px and AKAZE 0.78/5 px**. To inspect an existing run, leave `rerun_pipeline = False` and set `run_directory` to its path.

The full CLI uses the coarse grid above; use the notebook for the refinement grid reported in the PDF.

| Parameter | Setting |
| --- | --- |
| Feature budget | Up to 5,000 per image; stable response sorting keeps descriptor rows aligned |
| SIFT / ORB | `nfeatures=5000`; ORB `WTA_K=2` |
| AKAZE | `threshold=0.001`, `max_points=5000`; default oriented full MLDB |
| Matching | BF KNN k=2; L2 for SIFT, Hamming for ORB/AKAZE |
| RANSAC | Confidence 0.995; max 5,000 iterations |
| Common frame | Image 3; direct H for 1→3, 2→3, 4→3 |
| Canvas / main blending | 40 million-pixel limit; geometric support; premultiplied colour warp; distance-weighted feather blend; support crop |
| Common baseline | Ratio 0.75 / threshold 3 px |
| Sensitivity | Correspondence-order permutations with seeds 42, 123, 2026; saved main H retained |
| Blending comparison | Feather and multi-band 3/5/7 bands on the same selected H |
| Repeated comparison | 1 warmup + 5 samples, median (IQR), one OpenCV thread, OpenCL off |

Initial matches count two-neighbour KNN rows before the ratio test. Inlier ratio divides by ratio-pass matches. Pooled forward reprojection RMSE uses summed squared inlier errors divided by total inliers over the three source-to-anchor connections. Keypoint totals count the four inputs once each.

The repeated total time includes grayscale conversion, detector creation, features, matching, H, canvas, warp, feather blend and crop. The separate blending time covers canvas/warp/blend/crop only. Both exclude metrics, I/O and plotting. Grid times are single samples. Compare matching timing scopes rather than adding separately measured medians.

## 7. Recorded outputs

The report uses these fixed experiment IDs:

| Part | Run ID | Included evidence |
| --- | --- | --- |
| A | `20261019T024705Z_aef5c7be` | Summary/metric/timing CSVs, run and validation metadata, derived tables, and the seven figures referenced by the report |
| B | `20261019T103316Z_9fe8bb55` | Metric CSVs, camera/reference metadata, report figures, and raw/filtered SGBM point clouds |
| C, main | `20261019T105909Z_fe7aa242` | Metric/comparison CSVs, run metadata, and the figures referenced by the report |

The repository keeps the result files needed to read the reported measurements, inspect their manifests, display the figures, and compile the PDF. Large per-configuration images, duplicate figure exports, and intermediate feature/correspondence arrays are omitted; the Part A/B/C commands above regenerate full run directories with those artifacts. Each new run receives its own timestamp/UUID directory. When replacing a report-linked run, update its tables and figure paths together. Figures can be rebuilt with `build_figures` in the corresponding module.

### Main measured outcomes

| Part | Recorded result |
| --- | --- |
| A | Gaussian σₙ20: Gaussian 7×7/σ_f1.2 gives 31.2319 dB PSNR; SP 3%: Median 3×3 gives 37.8470 dB |
| B | SGBM n32 b5: 0.5341 px RMSE, 91.48% GT-domain coverage; common-mask n32 b9 comparison also favours SGBM |
| C | Selected total-time medians: SIFT 2838.351 ms, ORB 1025.075 ms, AKAZE 1965.790 ms; selected corner changes 1.565/24.328/6.038 px |

## 8. Report

Restore the recorded result directories listed above before compiling: the report includes their figures directly. With TeX Live and `latexmk` installed, run from the project root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error report_source/main.tex
```

The included `.latexmkrc` resolves paths from the source directory and writes `report_source/build/main.pdf`. Keep `part_a_full_results.tex` beside `main.tex`; it contains the 528 final-grid groups from Part A's recorded `summary.csv`. Re-running an experiment creates a new directory and does not update the report's fixed tables or figure paths automatically. When adopting new measurements, update the tables, appendix and figures together.

## 9. Known limitations

- **Part A:** Parameters were selected and evaluated on one reference image with three noise seeds. The results do not establish performance on unseen images. The photograph's acquisition source remains unconfirmed.
- **Part B:** The camera model is assumed, including horizontal FoV and normalized baseline. Point clouds illustrate relative geometry in normalized-baseline units; measured calibration and physical depth accuracy remain unverified. The flow-to-disparity conversion applies only to the verified horizontal Venus pair. No occlusion ground-truth mask is available.
- **Part C:** Comparisons use one four-image sequence. Planar homographies leave perspective stretching and some overlap ghosting; blending cannot recover saturated highlights. Reprojection RMSE and correspondence-order sensitivity do not establish absolute geometric accuracy.
- **Reproducibility:** Code-only checkouts omit the local inputs and recorded outputs. Supply these with the submission to inspect the recorded report; the Middlebury files can also be downloaded using the archive links above. New runtimes depend on the machine and environment, even with the same parameters and seeds.
