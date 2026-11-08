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


def benchmark_one(label, make_encode_command, input_path, qoaconv,
                  input_samples, input_rate, input_channels, repeats, temp_dir):
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
        subprocess.run(command, check=True)
        encode_times.append(time.perf_counter() - start)

        subprocess.run(
            [qoaconv, str(qoa_path), str(decoded_path)],
            check=True,
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

    return {
        "encoder": label,
        "seconds": statistics.median(encode_times),
        "psnr_db": statistics.mean(qualities),
    }


def pareto_frontier(points):
    """Return points not dominated by a faster-or-equal, higher-quality point."""
    frontier = []

    for point in points:
        dominated = any(
            other is not point
            and other["seconds"] <= point["seconds"]
            and other["psnr_db"] >= point["psnr_db"]
            and (
                other["seconds"] < point["seconds"]
                or other["psnr_db"] > point["psnr_db"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(point)

    return sorted(frontier, key=lambda point: point["seconds"])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark QOA encoder speed and quality, "
            "then plot the Pareto frontier."
        )
    )
    parser.add_argument(
        "wav",
        type=Path,
        help="Input 16-bit signed PCM WAV file",
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
        help="Runs per encoder; reports median time and mean PSNR (default: 3)",
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
        help="Output CSV results (default: qoa_benchmark.csv)",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    input_path = args.wav.resolve()
    qoamax = str(Path(args.qoamax).resolve())
    qoaconv = args.qoaconv
    if Path(qoaconv).exists():
        qoaconv = str(Path(qoaconv).resolve())

    input_samples, input_rate, input_channels = read_pcm16_wav(input_path)
    print(
        f"Input: {input_path} "
        f"({input_rate} Hz, {input_channels} channel(s))"
    )

    points = []
    with tempfile.TemporaryDirectory(prefix="qoa-benchmark-") as directory:
        temp_dir = Path(directory)

        for effort in EFFORTS:
            if effort in args.exclude_effort:
                print(f"Skipping qoamax --effort {effort}")
                continue

            print(f"Benchmarking qoamax --effort {effort}...")
            points.append(
                benchmark_one(
                    label=f"qoamax-{effort}",
                    make_encode_command=lambda output, effort=effort: [
                        qoamax,
                        "encode",
                        input_path,
                        output,
                        "--effort",
                        effort,
                    ],
                    input_path=input_path,
                    qoaconv=qoaconv,
                    input_samples=input_samples,
                    input_rate=input_rate,
                    input_channels=input_channels,
                    repeats=args.repeats,
                    temp_dir=temp_dir,
                )
            )

        print("Benchmarking reference qoaconv...")
        points.append(
            benchmark_one(
                label="qoaconv-reference",
                make_encode_command=lambda output: [
                    qoaconv,
                    input_path,
                    output,
                ],
                input_path=input_path,
                qoaconv=qoaconv,
                input_samples=input_samples,
                input_rate=input_rate,
                input_channels=input_channels,
                repeats=args.repeats,
                temp_dir=temp_dir,
            )
        )

    frontier = pareto_frontier(points)
    frontier_names = {point["encoder"] for point in frontier}

    with args.csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["encoder", "seconds", "psnr_db", "pareto"],
        )
        writer.writeheader()

        for point in points:
            writer.writerow({
                **point,
                "pareto": point["encoder"] in frontier_names,
            })

    fig, ax = plt.subplots(figsize=(9, 6))

    custom = [
        point for point in points
        if point["encoder"].startswith("qoamax-")
    ]
    reference = [
        point for point in points
        if point["encoder"] == "qoaconv-reference"
    ]

    ax.scatter(
        [point["seconds"] for point in custom],
        [point["psnr_db"] for point in custom],
        label="qoamax",
        marker="o",
    )
    ax.scatter(
        [point["seconds"] for point in reference],
        [point["psnr_db"] for point in reference],
        label="Reference",
        marker="s",
        color="tab:orange",
    )

    for point in points:
        label = point["encoder"].replace("qoamax-", "")
        ax.annotate(
            label,
            (point["seconds"], point["psnr_db"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    if frontier:
        ax.plot(
            [point["seconds"] for point in frontier],
            [point["psnr_db"] for point in frontier],
            "--",
            color="black",
            label="Pareto frontier",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Encoding wall time (seconds; lower is better)")
    ax.set_ylabel("PSNR (dB; higher is better)")
    ax.set_title("QOA encoder speed vs. quality")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.svg, format="svg")
    plt.close(fig)

    print("\nResults (median encode time, mean PSNR):")
    for point in sorted(points, key=lambda item: item["seconds"]):
        psnr = (
            "inf"
            if math.isinf(point["psnr_db"])
            else f'{point["psnr_db"]:.3f}'
        )
        marker = (
            "  [Pareto]"
            if point["encoder"] in frontier_names
            else ""
        )
        print(
            f'{point["encoder"]:20s} '
            f'{point["seconds"]:9.4f} s  '
            f'{psnr:>8s} dB{marker}'
        )

    print(f"\nWrote {args.svg} and {args.csv}")


if __name__ == "__main__":
    main()
