# INS3155 — Computer Vision Midterm

Traditional Image Processing for Filtering, 3D Reconstruction, and Image Stitching.

The implemented component is **Part A: Image Filtering and Enhancement**. It compares Mean, Gaussian and Median denoising on Gaussian and Salt-and-Pepper corruption, then evaluates Laplacian sharpening. Parts B and C have not been implemented.

## Files

| Path | Contents |
| --- | --- |
| `src/part_a.py` | Noise generation, filtering, sharpening, quality metrics, timing, validation and figure export |
| `notebooks/part_a.ipynb` | Experiment configuration, execution, tables and visual comparisons |
| `data/part_a/succulent_20231108_171619.jpg` | Original JPEG used for Part A |
| `data/part_a/metadata.json` | Image source, checksum and preprocessing |
| `results/part_a/official/` | Recorded full runs, each in a separate directory |
| `results/part_a/succulent/parameter_search/` | Parameter-search measurements and selection evidence |
| `report_source/part_a.md` | Separate Part A report artifact with measured tables and linked figures |
| `pyproject.toml`, `uv.lock` | Dependencies and locked environment |

## Installation

Use Python **3.13 or later** and [uv](https://docs.astral.sh/uv/). Run commands from the project root:

```bash
uv sync --locked --group dev
```

The runtime dependencies are OpenCV, NumPy, Pillow, scikit-image and Matplotlib. The development group includes JupyterLab and the Python kernel needed to run the notebook. `pyproject.toml` declares dependencies; `uv.lock` fixes the resolved versions.

## Input image

Place the supplied JPEG at `data/part_a/succulent_20231108_171619.jpg`. Its SHA-256 is:

```text
ee48060d10e9c8e2ed6829860c4ef25144928029b6ebb755deb1c2ae449e8937
```

The photograph shows succulent leaves, gravel and a glazed pot against a desk/background. It serves as the reference for synthetic noise generation and image-quality evaluation.

The source JPEG is 4128×3096 with EXIF orientation 6. The loader corrects orientation, converts to RGB and resizes with `INTER_AREA` to 768×1024. It does not crop the input or modify the source file. Three fixed diagnostic crops inspect the pot surface, leaf edges and gravel. Primary metrics use the full resized image.

The source image, recorded measurements, figures and report are supplied separately from this code-only repository. The notebook can create a fresh full run from the JPEG without the earlier parameter-search files. To view the recorded run and its report figures, also supply its result directory.

## Run Part A

Start JupyterLab:

```bash
uv run jupyter lab notebooks/part_a.ipynb
```

Select the environment's **Python 3 (ipykernel)** kernel, then run all cells in order. The notebook reads the recorded run `results/part_a/official/20261019T024705Z_aef5c7be` when it exists. It checks the input, parameters, saved image hashes and metrics before displaying the results. It also compares the current measurement code with the recorded source snapshot before using cached results. Formatting, figure rendering and environment-metadata checks may change; changes to image processing, metrics, timing or experiment configuration require a fresh run.

To repeat all measurements, set `rerun_experiments = True` in the run cell and run all cells. When the recorded directory is absent, the notebook automatically measures a fresh run. Each fresh run gets a new timestamp/UUID directory under `results/part_a/official/`; existing runs are preserved. The notebook exports figures and comparison tables, then adds the ROI-objective and selected-output tables.

A smaller command-line run checks the pipeline with the same real input:

```bash
uv run python src/part_a.py --profile smoke --output results/part_a/smoke
```

The smoke profile uses a 256-pixel long edge, one seed, one level per noise type and a reduced parameter grid. Its scores are not the full experiment's reported results.

## Full experiment settings

| Setting | Values |
| --- | --- |
| Gaussian noise standard deviation | 10, 20, 30 in 8-bit intensity units; mean 0 |
| Salt-and-Pepper density | 1%, 3%, 5%; approximately equal salt and pepper counts |
| Random seeds | 42, 43, 44 |
| Saved image seed | 42 |
| Mean/Median/Gaussian kernels | 3×3, 5×5, 7×7, 9×9, 11×11 |
| Gaussian filter sigma | 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6 |
| Laplacian apertures | 3, 5, 7; derivative scale `2 ** (4 - 2*k)` |
| Sharpening strength alpha | 0.1, 0.2, 0.25, 0.5 after denoising; 0.25 and 0.5 for direct controls |
| Runtime protocol | 2 warm-ups and 10 repeats per seed; one OpenCV thread; OpenCL off |

The earlier search evaluated 382 distinct denoiser configurations across five kernels and 22 Gaussian sigma values in coarse/fine searches. The full experiment uses the smaller closed comparison grid listed above. For each noise condition and denoiser, its PSNR-selected setting feeds the sharpening sweep. The six highest-PSNR complete recipes within the measured grid are:

| Corruption | Denoising | Laplacian aperture | Alpha |
| --- | --- | --- | --- |
| Gaussian σ=10 | Gaussian 5×5, sigma 0.8 | 7 | 0.1 |
| Gaussian σ=20 | Gaussian 7×7, sigma 1.2 | 7 | 0.1 |
| Gaussian σ=30 | Gaussian 9×9, sigma 1.5 | 7 | 0.1 |
| Salt-and-Pepper 1% | Median 3×3 | 5 | 0.25 |
| Salt-and-Pepper 3% | Median 3×3 | 5 | 0.25 |
| Salt-and-Pepper 5% | Median 3×3 | 5 | 0.2 |

## Outputs and interpretation

Each full run contains:

| File/directory | Contents |
| --- | --- |
| `run.json` | Input hash/size, source snapshot, configuration, environment and realized noise statistics |
| `reference.png` | Oriented and resized RGB reference |
| `images/` | Noisy, denoised and sharpened PNGs for seed 42 |
| `metrics.csv` | 1,584 seed-level quality records and per-case runtime summaries |
| `timings.csv` | 15,660 raw runtime samples with measurement scope |
| `summary.csv` | 528 configuration groups aggregated across seeds |
| `validation.json` | Image/CSV consistency check result |
| `figures/` | 33 PNG/PDF figure pairs, including labelled processing grids and sensitivity plots |
| `tables/` | Noisy baselines, denoiser comparisons, sharpening comparisons, direct controls and ROI metrics |
| `tables/objective_roi_comparison.csv` | Notebook-generated PSNR-versus-SSIM ROI comparison for seed 42 |
| `tables/recommended_outputs.csv` | Notebook-generated six selected recipes, with example PNG paths and checksums |

Quality is reported as mean ± sample standard deviation over three seeds. MSE uses float64 RGB differences, PSNR uses a peak value of 255, and zero MSE yields infinite PSNR. SSIM uses an explicit range of 255, RGB channel averaging, Gaussian window weights with sigma 1.5 and population covariance.

Runtime is median (IQR) in ms over 30 samples per configuration. It includes algorithm checks and array conversions, while excluding image I/O, noise generation, metrics and plotting. Denoising plus sharpening is timed as a complete chain. Noisy baselines have no filter runtime. Timing samples from separate measurement runs are not combined.

Running all notebook cells regenerates the figures and tables for a fresh run. For a cached run, it checks source compatibility, validates the measurements and regenerates figures and comparison tables, including the two notebook-derived tables. The [Part A report](report_source/part_a.md) documents the recorded full run; its tables and figure links refer to that run rather than automatically changing to a later run.

The settings optimize this image and the measured noise realizations. Smooth regions can dominate whole-image scores, and higher SSIM can coincide with lost gravel detail. Compare the fixed crops alongside the primary metrics. Runtime varies with hardware and system load. The Markdown report covers Part A; a combined LaTeX/PDF report has not been produced.
