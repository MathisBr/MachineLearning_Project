import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a directory of audio files to .npy or .npz for faster loading.")
    parser.add_argument("--input", required=True, help="Input dataset root directory")
    parser.add_argument("--output", required=True, help="Output directory for numpy files")
    parser.add_argument("--sr", type=int, default=22050, help="Target sample rate")
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Force mono conversion (recommended for classification)",
    )
    parser.add_argument(
        "--format",
        choices=["npy", "npz"],
        default="npy",
        help="Output format (.npy is fastest, .npz is compressed)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "int16"],
        default="float32",
        help="Stored waveform dtype",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of workers (0 = auto)",
    )
    return parser.parse_args()


def list_audio_files(root: Path, exts: Iterable[str]) -> List[Path]:
    exts_l = {e.lower() for e in exts}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_l]
    files.sort()
    return files


def load_audio(path: Path, target_sr: int, mono: bool) -> Tuple[np.ndarray, int]:
    # librosa can decode many formats (wav/mp3/flac/ogg/...) and resample directly.
    import librosa

    y, sr = librosa.load(path.as_posix(), sr=target_sr, mono=mono)
    return y, sr


def cast_waveform(y: np.ndarray, out_dtype: str) -> np.ndarray:
    if out_dtype == "float32":
        return y.astype(np.float32, copy=False)
    if out_dtype == "float16":
        return y.astype(np.float16)
    # int16 is smaller and fast to load; values are scaled from [-1, 1] to int16 range.
    y_clip = np.clip(y, -1.0, 1.0)
    return (y_clip * 32767.0).astype(np.int16)


def convert_one(
    in_file: Path,
    input_root: Path,
    output_root: Path,
    sr: int,
    mono: bool,
    out_format: str,
    out_dtype: str,
) -> Dict[str, str]:
    rel = in_file.relative_to(input_root)
    base_out = output_root / rel.with_suffix("")
    out_file = base_out.with_suffix(f".{out_format}")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    y, used_sr = load_audio(in_file, target_sr=sr, mono=mono)
    y = cast_waveform(y, out_dtype)

    if out_format == "npy":
        np.save(out_file.as_posix(), y)
    else:
        np.savez_compressed(out_file.as_posix(), audio=y, sr=np.array([used_sr], dtype=np.int32))

    return {
        "input": str(in_file),
        "output": str(out_file),
        "samples": str(y.shape[-1] if y.ndim > 0 else 0),
        "sr": str(used_sr),
        "dtype": str(y.dtype),
    }


def main() -> None:
    args = parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists() or not input_root.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_root}")

    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
    files = list_audio_files(input_root, audio_exts)
    if not files:
        print("No audio files found. Nothing converted.")
        return

    workers = args.workers if args.workers and args.workers > 0 else None

    manifest: List[Dict[str, str]] = []
    failures: List[Dict[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                convert_one,
                f,
                input_root,
                output_root,
                args.sr,
                args.mono,
                args.format,
                args.dtype,
            ): f
            for f in files
        }

        total = len(futures)
        done = 0

        for fut in concurrent.futures.as_completed(futures):
            src = futures[fut]
            done += 1
            try:
                item = fut.result()
                manifest.append(item)
                if done % 200 == 0 or done == total:
                    print(f"Converted {done}/{total}")
            except Exception as exc:  # noqa: BLE001
                failures.append({"input": str(src), "error": str(exc)})

    manifest.sort(key=lambda x: x["input"])

    manifest_file = output_root / "manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_root": str(input_root),
                "output_root": str(output_root),
                "sample_rate": args.sr,
                "mono": args.mono,
                "format": args.format,
                "dtype": args.dtype,
                "num_converted": len(manifest),
                "num_failed": len(failures),
                "items": manifest,
                "failures": failures,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Done. Converted: {len(manifest)} | Failed: {len(failures)}")
    print(f"Manifest: {manifest_file}")


if __name__ == "__main__":
    main()
