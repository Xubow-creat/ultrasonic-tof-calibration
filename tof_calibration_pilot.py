"""Small, reproducible k-Wave pilot for calibration-constrained TFM.

This deliberately limited experiment tests whether known sparse reflectors can
identify a layer velocity and improve focusing of an unseen reflector. It is
2-D scalar acoustics; it does not model elastic weld anisotropy or full NDT.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import correlate

ROOT = Path(__file__).resolve().parent
try:
    import kwave
except ModuleNotFoundError:
    # Local development checkout; a published clone should install the pinned
    # PyPI dependency instead of copying the third-party package into this repo.
    local_kwave = ROOT.parent / "k-wave-python-master" / "k-wave-python-master"
    if not local_kwave.is_dir():
        raise
    sys.path.insert(0, str(local_kwave))
    import kwave

from kwave.data import Vector  # noqa: E402
from kwave.kgrid import kWaveGrid  # noqa: E402
from kwave.kmedium import kWaveMedium  # noqa: E402
from kwave.ksensor import kSensor  # noqa: E402
from kwave.ksource import kSource  # noqa: E402
from kwave.kspaceFirstOrder import kspaceFirstOrder  # noqa: E402


@dataclass(frozen=True)
class Config:
    dx_mm: float = 0.12
    nx: int = 156
    nz: int = 132
    n_elements: int = 8
    pitch_mm: float = 0.72
    source_z_mm: float = 1.8
    layer_bottom_mm: float = 4.8
    layer_speed_m_s: float = 4500.0
    bulk_speed_m_s: float = 5900.0
    layer_density: float = 7400.0
    bulk_density: float = 7850.0
    lens_delta_m_s: float = 0.0
    lens_center_mm: float = 1.6
    lens_width_mm: float = 1.8
    pulse_mhz: float = 2.2
    source_time_us: float = 0.65
    t_end_us: float = 6.5
    pml_cells: int = 10
    inclusion_sigma_mm: float = 0.18
    inclusion_strength: float = 0.35


CALIBRATORS = ((-1.8, 8.0), (1.8, 10.0))
TARGET = (1.1, 11.4)


def ricker(t: np.ndarray, f0: float, t0: float) -> np.ndarray:
    a = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * a * a) * np.exp(-a * a)


def coordinates(cfg: Config):
    x = (np.arange(cfg.nx) - cfg.nx // 2) * cfg.dx_mm
    z = np.arange(cfg.nz) * cfg.dx_mm
    elements = (np.arange(cfg.n_elements) - (cfg.n_elements - 1) / 2) * cfg.pitch_mm
    return x, z, elements


def build_run(cfg: Config, tx: int, inclusion: tuple[float, float] | None):
    x, z, elements = coordinates(cfg)
    grid = kWaveGrid(Vector([cfg.nz, cfg.nx]), Vector([cfg.dx_mm * 1e-3] * 2))
    dt = 0.23 * cfg.dx_mm * 1e-3 / cfg.bulk_speed_m_s
    nt = int(np.ceil(cfg.t_end_us * 1e-6 / dt))
    grid.setTime(nt, dt)
    c = np.full((cfg.nz, cfg.nx), cfg.bulk_speed_m_s, dtype=np.float32)
    rho = np.full_like(c, cfg.bulk_density)
    c[z < cfg.layer_bottom_mm, :] = cfg.layer_speed_m_s
    rho[z < cfg.layer_bottom_mm, :] = cfg.layer_density
    if cfg.lens_delta_m_s:
        lateral = cfg.lens_delta_m_s * np.exp(
            -0.5 * ((x - cfg.lens_center_mm) / cfg.lens_width_mm) ** 2)
        c[z < cfg.layer_bottom_mm, :] += lateral[None, :]
    if inclusion is not None:
        cx, cz = inclusion
        xx, zz = np.meshgrid(x, z)
        blob = np.exp(-((xx - cx) ** 2 + (zz - cz) ** 2) /
                      (2.0 * cfg.inclusion_sigma_mm**2))
        c *= (1.0 - cfg.inclusion_strength * blob).astype(np.float32)
        rho *= (1.0 - cfg.inclusion_strength * blob).astype(np.float32)
    medium = kWaveMedium(sound_speed=c, density=rho)
    iz = int(np.argmin(abs(z - cfg.source_z_mm)))
    ix = np.array([np.argmin(abs(x - e)) for e in elements])
    source = kSource()
    source.p_mask = np.zeros_like(c, dtype=bool)
    source.p_mask[iz, ix[tx]] = True
    t = np.asarray(grid.t_array).ravel()
    source.p = (5e4 * ricker(t, cfg.pulse_mhz * 1e6,
                            cfg.source_time_us * 1e-6)).astype(np.float32)[None, :]
    sensor = kSensor(mask=np.zeros_like(c, dtype=bool))
    sensor.mask[iz, ix] = True
    sensor.record = ["p"]
    return grid, medium, source, sensor, t, elements


def run_fmc(cfg: Config, inclusion: tuple[float, float] | None):
    traces = []
    binary_name = "kspaceFirstOrder-OMP.exe" if platform.system() == "Windows" else "kspaceFirstOrder-OMP"
    binary = Path(kwave.__file__).resolve().parent / "bin" / platform.system().lower() / binary_name
    for tx in range(cfg.n_elements):
        grid, medium, source, sensor, t, elements = build_run(cfg, tx, inclusion)
        result = kspaceFirstOrder(
            grid, medium, source, sensor,
            backend="cpp", device="cpu", pml_inside=True,
            pml_size=cfg.pml_cells, quiet=True, num_threads=4,
            binary_path=str(binary) if binary.is_file() else None,
        )
        p = np.asarray(result["p"], dtype=np.float32)
        if p.shape[0] != cfg.n_elements:
            p = p.T
        assert p.shape == (cfg.n_elements, len(t)), p.shape
        traces.append(p)
    return np.stack(traces), t, elements


def one_way_layered(element_x: float, x: float, z: float,
                    cfg: Config, layer_speed_m_s: float) -> float:
    """Fermat path through one flat interface; return microseconds."""
    a = cfg.layer_bottom_mm - cfg.source_z_mm
    b = z - cfg.layer_bottom_mm
    c1 = layer_speed_m_s * 1e-3
    c2 = cfg.bulk_speed_m_s * 1e-3
    if b <= 0:
        return np.hypot(x - element_x, z - cfg.source_z_mm) / c1
    if abs(x - element_x) < 1e-12:
        return a / c1 + b / c2
    left, right = sorted((element_x, x))
    result = minimize_scalar(
        lambda xi: np.hypot(xi - element_x, a) / c1
        + np.hypot(x - xi, b) / c2,
        bounds=(left, right), method="bounded",
        options={"xatol": 1e-7},
    )
    return float(result.fun)


def one_way_global(element_x: float, x: float, z: float,
                   cfg: Config, speed_m_s: float) -> float:
    return np.hypot(x - element_x, z - cfg.source_z_mm) / (speed_m_s * 1e-3)


def one_way_lens(element_x: float, x: float, z: float, cfg: Config,
                 base_speed_m_s: float, delta_m_s: float,
                 center_mm: float | None = None) -> float:
    """Two-segment ray with slowness integration through a lateral velocity lens."""
    a = cfg.layer_bottom_mm - cfg.source_z_mm
    b = z - cfg.layer_bottom_mm
    nodes = (np.arange(9) + 0.5) / 9
    if center_mm is None:
        center_mm = cfg.lens_center_mm

    def top_time(x_end):
        xs = element_x + nodes * (x_end - element_x)
        c = base_speed_m_s + delta_m_s * np.exp(
            -0.5 * ((xs - center_mm) / cfg.lens_width_mm) ** 2)
        return np.hypot(x_end - element_x, a) * np.mean(1.0 / c) * 1000.0

    if b <= 0:
        return top_time(x)
    lo, hi = sorted((element_x, x))
    if hi - lo < 1e-12:
        return top_time(x) + b / (cfg.bulk_speed_m_s * 1e-3)
    result = minimize_scalar(
        lambda xi: top_time(xi)
        + np.hypot(x - xi, b) / (cfg.bulk_speed_m_s * 1e-3),
        bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    return float(result.fun)


def matched_arrivals(fmc: np.ndarray, t: np.ndarray, point,
                     elements: np.ndarray, cfg: Config):
    """Pick the strongest signed-wavelet match around the nominal reflector time."""
    dt_us = (t[1] - t[0]) * 1e6
    template_t = np.arange(-0.5, 0.5 + dt_us, dt_us) * 1e-6
    template = ricker(template_t, cfg.pulse_mhz * 1e6, 0.0)
    picks = np.full((cfg.n_elements, cfg.n_elements), np.nan)
    quality = np.zeros_like(picks)
    for i, xi in enumerate(elements):
        for j, xj in enumerate(elements):
            expected = (one_way_global(xi, *point, cfg, cfg.bulk_speed_m_s)
                        + one_way_global(xj, *point, cfg, cfg.bulk_speed_m_s)
                        + cfg.source_time_us)
            corr = correlate(fmc[i, j], template, mode="same", method="fft")
            lo = max(0, int((expected - 0.75) / dt_us))
            hi = min(len(t), int((expected + 1.4) / dt_us))
            if hi <= lo:
                continue
            peak = lo + int(np.argmax(np.abs(corr[lo:hi])))
            picks[i, j] = t[peak] * 1e6
            quality[i, j] = abs(corr[peak])
    return picks, quality


def fit_model(picks, quality, elements, cfg: Config, model: str):
    rows = []
    for point, p, q in zip(CALIBRATORS, picks, quality):
        q_max = max(float(np.max(q)), 1e-12)
        for i in range(cfg.n_elements):
            for j in range(i, cfg.n_elements):
                if np.isfinite(p[i, j]) and q[i, j] > 0.08 * q_max:
                    rows.append((point, i, j, p[i, j], q[i, j] / q_max))
    if len(rows) < 12:
        raise RuntimeError(f"Only {len(rows)} usable calibration arrivals")

    def one_way(e, x, z, speed, delta, center):
        if model == "layered":
            return one_way_layered(e, x, z, cfg, speed)
        if model in ("lens", "lens_free_center"):
            return one_way_lens(e, x, z, cfg, speed, delta, center)
        return one_way_global(e, x, z, cfg, speed)

    # Include a common pulse/reflector timing offset as a nuisance parameter.
    def residual(params):
        if model == "lens_free_center":
            speed, delta, center, offset = params
        elif model == "lens":
            speed, delta, offset = params
            center = cfg.lens_center_mm
        else:
            speed, offset = params
            delta = 0.0
            center = cfg.lens_center_mm
        out = []
        cache = {}
        for (x, z), i, j, measured, weight in rows:
            for index in (i, j):
                key = (x, z, index)
                if key not in cache:
                    cache[key] = one_way(elements[index], x, z, speed, delta, center)
            predicted = cache[(x, z, i)] + cache[(x, z, j)] + offset
            out.append((predicted - measured) * np.sqrt(weight))
        return out

    if model == "lens_free_center":
        bounds = ([3500.0, -2000.0, -2.5, -0.5],
                  [6000.0, 0.0, 3.5, 1.5])
        x0 = [4500.0, -600.0, 0.0, cfg.source_time_us]
    elif model == "lens":
        bounds = ([3500.0, -2000.0, -0.5], [6000.0, 0.0, 1.5])
        x0 = [4500.0, -600.0, cfg.source_time_us]
    else:
        bounds = ([3500.0, -0.5], [6500.0, 1.5])
        x0 = [4800.0, cfg.source_time_us] if model == "layered" else [5400.0, cfg.source_time_us]
    result = least_squares(residual, x0=x0, bounds=bounds,
                           xtol=1e-8, ftol=1e-8, gtol=1e-8)
    return {"speed_m_s": float(result.x[0]),
            "lens_delta_m_s": float(result.x[1]) if model in ("lens", "lens_free_center") else 0.0,
            "lens_center_mm": float(result.x[2]) if model == "lens_free_center" else cfg.lens_center_mm,
            "offset_us": float(result.x[-1]),
            "calibration_rms_us": float(np.sqrt(np.mean(np.square(result.fun)))),
            "n_arrivals": len(rows)}


def tfm(fmc, t, elements, cfg, x_axis, z_axis, model, speed, offset,
        delta=0.0, center=None):
    dt_us = (t[1] - t[0]) * 1e6
    image = np.empty((len(z_axis), len(x_axis)), dtype=np.float32)
    for iz, z in enumerate(z_axis):
        for ix, x in enumerate(x_axis):
            if model == "layered":
                one = np.array([one_way_layered(e, x, z, cfg, speed) for e in elements])
            elif model in ("lens", "lens_free_center"):
                one = np.array([one_way_lens(e, x, z, cfg, speed, delta, center) for e in elements])
            else:
                one = np.array([one_way_global(e, x, z, cfg, speed) for e in elements])
            times = one[:, None] + one[None, :] + offset
            indices = times / dt_us
            lower = np.floor(indices).astype(int)
            valid = (lower >= 0) & (lower + 1 < fmc.shape[-1])
            lo = np.clip(lower, 0, fmc.shape[-1] - 2)
            i, j = np.indices(lo.shape)
            samples = fmc[i, j, lo] * (1 - (indices - lower)) \
                + fmc[i, j, lo + 1] * (indices - lower)
            image[iz, ix] = abs(np.sum(samples * valid))
    return image


def peak_stats(image, x_axis, z_axis, target):
    # Search the same fixed image region for every target and method.
    iz, ix = np.unravel_index(np.argmax(image), image.shape)
    pos = (float(x_axis[ix]), float(z_axis[iz]))
    return {"peak_x_mm": pos[0], "peak_z_mm": pos[1],
            "center_error_mm": float(np.hypot(pos[0] - target[0], pos[1] - target[1])),
            "peak_amplitude": float(image[iz, ix])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path.cwd() / "outputs" / "tof_calibration_pilot")
    parser.add_argument("--lens-delta", type=float, default=0.0,
                        help="Velocity perturbation in m/s; 0 gives a uniform layer")
    parser.add_argument("--target-x", type=float, default=TARGET[0])
    parser.add_argument("--target-z", type=float, default=TARGET[1])
    parser.add_argument("--reuse-calibration", type=Path,
                        help="Prior pilot_data.npz with the same medium and calibrators")
    parser.add_argument("--reuse-target", type=Path,
                        help="Prior pilot_data.npz with the same target and medium")
    parser.add_argument("--assumed-lens-center", type=float)
    parser.add_argument("--assumed-lens-width", type=float)
    parser.add_argument("--fit-lens-center", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = Config(lens_delta_m_s=args.lens_delta)
    inference_cfg = replace(
        cfg,
        lens_center_mm=(args.assumed_lens_center if args.assumed_lens_center is not None
                        else cfg.lens_center_mm),
        lens_width_mm=(args.assumed_lens_width if args.assumed_lens_width is not None
                       else cfg.lens_width_mm),
    )
    target = (args.target_x, args.target_z)
    start = time.perf_counter()
    if args.reuse_calibration:
        previous_report = json.loads(args.reuse_calibration.with_name("results.json").read_text())
        if previous_report["config"]["lens_delta_m_s"] != cfg.lens_delta_m_s:
            raise ValueError("Calibration medium does not match --lens-delta")
        previous = np.load(args.reuse_calibration)
        reference = previous["background_fmc"]
        t = previous["t_s"]
        elements = previous["elements_mm"]
        calibration_fmc = list(previous["calibration_fmc"])
    else:
        print("Running reference FMC", flush=True)
        reference, t, elements = run_fmc(cfg, None)
        calibration_fmc = []
        for point in CALIBRATORS:
            print(f"Running calibration FMC at {point}", flush=True)
            signal, _, _ = run_fmc(cfg, point)
            calibration_fmc.append(signal - reference)
    if args.reuse_target:
        target_report = json.loads(args.reuse_target.with_name("results.json").read_text())
        if tuple(target_report["held_out_target_mm"]) != target:
            raise ValueError("Reused target location does not match --target-x/--target-z")
        target_fmc = np.load(args.reuse_target)["target_fmc"]
    else:
        print(f"Running held-out target FMC at {target}", flush=True)
        target_raw, _, _ = run_fmc(cfg, target)
        target_fmc = target_raw - reference
    picks, quality = zip(*(matched_arrivals(fmc, t, p, elements, cfg)
                           for fmc, p in zip(calibration_fmc, CALIBRATORS)))
    layered = fit_model(picks, quality, elements, inference_cfg, "layered")
    global_fit = fit_model(picks, quality, elements, inference_cfg, "global")
    lens_model = "lens_free_center" if args.fit_lens_center else "lens"
    lens_fit = fit_model(picks, quality, elements, inference_cfg, lens_model) if cfg.lens_delta_m_s else None
    x_axis = np.arange(-3.6, 3.61, 0.1)
    z_axis = np.arange(8.2, 14.41, 0.1)
    results = {}
    images = {}
    methods = [
        ("nominal", "global", cfg.bulk_speed_m_s, cfg.source_time_us, 0.0, None),
        ("global_calibrated", "global", global_fit["speed_m_s"], global_fit["offset_us"], 0.0, None),
        ("layered_calibrated", "layered", layered["speed_m_s"], layered["offset_us"], 0.0, None),
    ]
    if lens_fit:
        methods.extend([
            ("lens_calibrated", lens_model, lens_fit["speed_m_s"],
             lens_fit["offset_us"], lens_fit["lens_delta_m_s"],
             lens_fit["lens_center_mm"]),
            ("lens_true_speed_assumed_shape", "lens", cfg.layer_speed_m_s,
             lens_fit["offset_us"], cfg.lens_delta_m_s, inference_cfg.lens_center_mm),
        ])
    else:
        methods.append(("layered_oracle", "layered", cfg.layer_speed_m_s,
                        layered["offset_us"], 0.0, None))
    for name, model, speed, offset, delta, center in methods:
        image = tfm(target_fmc, t, elements, inference_cfg, x_axis, z_axis,
                    model, speed, offset, delta, center)
        images[name] = image
        results[name] = peak_stats(image, x_axis, z_axis, target)
    report = {
        "scope": "2D acoustic k-Wave, reference-subtracted FMC; diagnostic pilot only",
        "config": asdict(cfg), "assumed_lens_center_mm": inference_cfg.lens_center_mm,
        "assumed_lens_width_mm": inference_cfg.lens_width_mm,
        "calibrators_mm": CALIBRATORS,
        "held_out_target_mm": target, "layered_fit": layered,
        "global_fit": global_fit, "lens_fit": lens_fit, "imaging": results,
        "lens_center_fitted": args.fit_lens_center,
        "wall_seconds": round(time.perf_counter() - start, 2),
    }
    np.savez_compressed(args.out / "pilot_data.npz", t_s=t, elements_mm=elements,
                        background_fmc=reference, target_fmc=target_fmc,
                        calibration_fmc=np.stack(calibration_fmc),
                        calibration_picks_us=np.stack(picks),
                        x_axis_mm=x_axis, z_axis_mm=z_axis,
                        **{f"image_{k}": v for k, v in images.items()})
    (args.out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
