#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save the SVG without opening a window.
import matplotlib.pyplot as plt
import numpy as np


EFFORTS = [
    "ultrafast",
    "superfast",
    "veryfast",
    "fast",
    "medium",
    "slow",
    "veryslow",
    "superslow",
    "placebo",
]


def read_pcm16_wav(path):
    """Return WAV samples as an (frames, channels) int16 array, plus metadata."""
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError(f"{path}: expected uncompressed PCM WAV")
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM WAV")

        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
    return samples, sample_rate, channels


def psnr_db(reference, decoded):
    if reference.shape != decoded.shape:
        raise ValueError(
            f"Decoded audio shape {decoded.shape} does not match "
            f"input shape {reference.shape}"
        )

    difference = reference.astype(np.float64) - decoded.astype(np.float64)
    mse = np.mean(difference * difference)

    if mse == 0:
        return math.inf

    # Full-scale peak for signed 16-bit PCM.
    return 20.0 * math.log10(32768.0 / math.sqrt(mse))


def benchmark_one(label, make_encode_command, qoaconv, input_samples,
                  input_rate, input_channels, repeats, temp_dir):
    """Encode one file `repeats` times; return the median time and the PSNR."""
    encode_times = []
    qualities = []

    for run_number in range(repeats):
        qoa_path = temp_dir / f"{label}_{run_number}.qoa"
        decoded_path = temp_dir / f"{label}_{run_number}_decoded.wav"

        command = [
            str(arg) if isinstance(arg, Path) else arg
            for arg in make_encode_command(qoa_path)
        ]

        # Time the encoder only; decoding and PSNR calculation are excluded.
        start = time.perf_counter()
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        encode_times.append(time.perf_counter() - start)

        subprocess.run(
            [qoaconv, str(qoa_path), str(decoded_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        decoded_samples, decoded_rate, decoded_channels = read_pcm16_wav(
            decoded_path
        )
        if decoded_rate != input_rate or decoded_channels != input_channels:
            raise ValueError(
                f"{label}: decoded WAV format differs from input "
                f"({decoded_rate} Hz, {decoded_channels} channels; expected "
                f"{input_rate} Hz, {input_channels} channels)"
            )

        qualities.append(psnr_db(input_samples, decoded_samples))
        qoa_path.unlink()
        decoded_path.unlink()

    # The encoders are deterministic, so every run has the same PSNR.
    return statistics.median(encode_times), statistics.mean(qualities)


def summarize(label, per_file):
    """Combine one encoder's per-file results into a single point.

    `gain_db` is the PSNR difference to qoaconv on the same file. Its mean
    weights every file equally, regardless of its length. Files where either
    encoder is lossless (infinite PSNR) have no finite gain and are skipped.
    """
    gains = [
        result["gain_db"] for result in per_file
        if math.isfinite(result["gain_db"])
    ]
    seconds = sum(result["seconds"] for result in per_file)
    duration = sum(result["duration"] for result in per_file)
    return {
        "encoder": label,
        "seconds": seconds,
        "realtime": duration / seconds if seconds > 0 else math.inf,
        "gain_db": statistics.mean(gains) if gains else math.nan,
        "min_gain_db": min(gains) if gains else math.nan,
        "max_gain_db": max(gains) if gains else math.nan,
        "files": len(gains),
    }


def pareto_frontier(points):
    """Return points not dominated by a faster-or-equal, higher-quality point."""
    frontier = []

    for point in points:
        dominated = any(
            other is not point
            and other["seconds"] <= point["seconds"]
            and other["gain_db"] >= point["gain_db"]
            and (
                other["seconds"] < point["seconds"]
                or other["gain_db"] > point["gain_db"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(point)

    return sorted(frontier, key=lambda point: point["seconds"])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark QOA encoder speed and quality on one or more files, "
            "then plot the Pareto frontier. Quality is the mean PSNR gain "
            "over the reference encoder qoaconv (0 dB) across the files."
        )
    )
    parser.add_argument(
        "wav",
        type=Path,
        nargs="+",
        help="Input 16-bit signed PCM WAV files",
    )
    parser.add_argument(
        "--qoamax",
        default="build/qoamax",
        help="Path to qoamax (default: build/qoamax)",
    )
    parser.add_argument(
        "--qoaconv",
        default="qoaconv",
        help="Path or command name for qoaconv (default: qoaconv)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Runs per encoder and file; uses the median time (default: 3)",
    )
    parser.add_argument(
        "--exclude-effort",
        action="append",
        choices=EFFORTS,
        default=[],
        help="Effort level to skip; may be used more than once",
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=Path("qoa_pareto.svg"),
        help="Output SVG plot (default: qoa_pareto.svg)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("qoa_benchmark.csv"),
        help="Output CSV with the combined results (default: qoa_benchmark.csv)",
    )
    parser.add_argument(
        "--per-file-csv",
        type=Path,
        help="Also write every encoder's result per file to this CSV",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    qoamax = str(Path(args.qoamax).resolve())
    qoaconv = args.qoaconv
    if Path(qoaconv).exists():
        qoaconv = str(Path(qoaconv).resolve())

    efforts = [effort for effort in EFFORTS if effort not in args.exclude_effort]
    for effort in args.exclude_effort:
        print(f"Skipping qoamax --effort {effort}")

    # (label, function returning the encode command for an input and output)
    encoders = [(
        "qoaconv-reference",
        lambda input_path, output: [qoaconv, input_path, output],
    )] + [(
        f"qoamax-{effort}",
        lambda input_path, output, effort=effort: [
            qoamax, "encode", "-q", input_path, output, "--effort", effort,
        ],
    ) for effort in efforts]

    per_file = {label: [] for label, _ in encoders}
    with tempfile.TemporaryDirectory(prefix="qoa-benchmark-") as directory:
        temp_dir = Path(directory)

        for file_number, wav in enumerate(args.wav, 1):
            input_path = wav.resolve()
            input_samples, input_rate, input_channels = read_pcm16_wav(
                input_path
            )
            duration = len(input_samples) / input_rate
            print(
                f"[{file_number}/{len(args.wav)}] {input_path} "
                f"({input_rate} Hz, {input_channels} channel(s), "
                f"{duration:.1f} s)"
            )

            reference_psnr = None
            for label, make_command in encoders:
                print(f"  {label}...", end="", flush=True)
                seconds, psnr = benchmark_one(
                    label=label,
                    make_encode_command=lambda output, make_command=make_command:
                        make_command(input_path, output),
                    qoaconv=qoaconv,
                    input_samples=input_samples,
                    input_rate=input_rate,
                    input_channels=input_channels,
                    repeats=args.repeats,
                    temp_dir=temp_dir,
                )
                if reference_psnr is None:
                    reference_psnr = psnr  # qoaconv runs first
                if math.isinf(psnr) or math.isinf(reference_psnr):
                    gain = math.inf if psnr > reference_psnr else math.nan
                else:
                    gain = psnr - reference_psnr
                print(f" {seconds:.3f} s, {psnr:.3f} dB ({gain:+.3f} dB)")
                per_file[label].append({
                    "encoder": label,
                    "file": str(wav),
                    "duration": duration,
                    "seconds": seconds,
                    "psnr_db": psnr,
                    "gain_db": gain,
                })

    points = [summarize(label, results) for label, results in per_file.items()]
    if points[0]["files"] < len(args.wav):
        print(
            f"Warning: {len(args.wav) - points[0]['files']} lossless file(s) "
            "excluded from the mean gain"
        )

    frontier = pareto_frontier(
        [point for point in points if math.isfinite(point["gain_db"])]
    )
    frontier_names = {point["encoder"] for point in frontier}

    with args.csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "encoder", "seconds", "realtime", "gain_db", "min_gain_db",
                "max_gain_db", "files", "pareto",
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow({
                **point,
                "pareto": point["encoder"] in frontier_names,
            })

    if args.per_file_csv:
        with args.per_file_csv.open("w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "encoder", "file", "duration", "seconds", "psnr_db",
                    "gain_db",
                ],
            )
            writer.writeheader()
            for results in per_file.values():
                writer.writerows(results)

    files = len(args.wav)
    fig, ax = plt.subplots(figsize=(9, 6))

    custom = [
        point for point in points
        if point["encoder"].startswith("qoamax-")
    ]
    reference = [
        point for point in points
        if point["encoder"] == "qoaconv-reference"
    ]

    ax.axhline(0, color="tab:orange", linewidth=0.8, alpha=0.5)
    # Error bars: the range of the per-file gains (smallest to largest).
    ax.errorbar(
        [point["seconds"] for point in custom],
        [point["gain_db"] for point in custom],
        yerr=[
            [point["gain_db"] - point["min_gain_db"] for point in custom],
            [point["max_gain_db"] - point["gain_db"] for point in custom],
        ],
        fmt="none",
        ecolor="tab:blue",
        alpha=0.5,
        capsize=3,
        label="Range over files" if files > 1 else None,
    )
    ax.scatter(
        [point["seconds"] for point in custom],
        [point["gain_db"] for point in custom],
        label="qoamax",
        marker="o",
    )
    ax.scatter(
        [point["seconds"] for point in reference],
        [point["gain_db"] for point in reference],
        label="Reference (qoaconv)",
        marker="s",
        color="tab:orange",
    )

    for point in points:
        label = point["encoder"].replace("qoamax-", "")
        ax.annotate(
            label,
            (point["seconds"], point["gain_db"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    if frontier:
        ax.plot(
            [point["seconds"] for point in frontier],
            [point["gain_db"] for point in frontier],
            "--",
            color="black",
            label="Pareto frontier",
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        "Encoding wall time"
        + (f", total of {files} files" if files > 1 else "")
        + " (seconds; lower is better)"
    )
    ax.set_ylabel(
        "PSNR gain over qoaconv"
        + (f", mean of {files} files" if files > 1 else "")
        + " (dB; higher is better)"
    )
    ax.set_title("QOA encoder speed vs. quality")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.svg, format="svg")
    plt.close(fig)

    print(
        f"\nResults over {files} file(s) (total of median encode times, "
        "PSNR gain over qoaconv: mean [min, max]):"
    )
    for point in sorted(points, key=lambda item: item["seconds"]):
        marker = (
            "  [Pareto]"
            if point["encoder"] in frontier_names
            else ""
        )
        print(
            f'{point["encoder"]:20s} '
            f'{point["seconds"]:9.3f} s {point["realtime"]:8.1f}x  '
            f'{point["gain_db"]:+7.3f} dB  '
            f'[{point["min_gain_db"]:+.3f}, {point["max_gain_db"]:+.3f}]'
            f'{marker}'
        )

    written = [args.svg, args.csv] + (
        [args.per_file_csv] if args.per_file_csv else []
    )
    print("\nWrote " + ", ".join(str(path) for path in written))


if __name__ == "__main__":
    main()
