"""Bounded-memory audio analysis. Originals are opened read-only by FFmpeg."""

from array import array
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

SAMPLE_RATE = 8000
POINTS = 720
MAX_FILE_BYTES = 500 * 1024 * 1024


def ffmpeg_binary():
    override = os.environ.get("WAVEFORM_FFMPEG")
    if override:
        return override
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def analyze(path, timeout=300):
    """Return a 720-point peak/RMS envelope, using bounded PCM chunks.

    Keep channel amplitudes independent so opposite-phase stereo doesn't
    disappear. Compact older analysis buckets adaptively to bound RAM even
    for long recordings. The 8 kHz analysis stream never replaces playback.
    """
    path = Path(path).resolve(strict=True)
    if not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("Audio must fit the existing 500 MiB upload limit")
    command = [ffmpeg_binary(), "-nostdin", "-v", "error", "-xerror",
               "-threads", "1", "-protocol_whitelist", "file,pipe",
               "-i", str(path), "-map", "0:a:0", "-vn", "-sn", "-dn",
               "-ac", "2", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    buckets = []
    bucket_frames = 128
    frames = 0
    peak = energy = 0.0
    count = 0
    expired = threading.Event()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        def expire():
            expired.set()
            process.kill()
        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
        try:
            remainder = b""
            while True:
                chunk = process.stdout.read(65536)
                if not chunk:
                    break
                chunk = remainder + chunk
                usable = len(chunk) - len(chunk) % 8
                remainder = chunk[usable:]
                samples = array("f")
                samples.frombytes(chunk[:usable])
                if sys.byteorder != "little":
                    samples.byteswap()
                for i in range(0, len(samples), 2):
                    left, right = samples[i], samples[i + 1]
                    if not math.isfinite(left) or not math.isfinite(right):
                        raise ValueError("Audio contains non-finite samples")
                    peak = max(peak, abs(left), abs(right))
                    energy += left * left + right * right
                    count += 1
                    frames += 1
                    if count == bucket_frames:
                        buckets.append((peak, energy, count))
                        peak = energy = 0.0
                        count = 0
                        if len(buckets) >= 8192:
                            buckets = [(max(a[0], b[0]), a[1] + b[1], a[2] + b[2])
                                       for a, b in zip(buckets[::2], buckets[1::2])]
                            bucket_frames *= 2
            code = process.wait()
            if expired.is_set():
                raise TimeoutError("Audio analysis exceeded its processing deadline")
            if code or remainder or not frames:
                raise ValueError("Audio could not be decoded")
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
                process.wait()
    if count:
        buckets.append((peak, energy, count))
    # Distribute each bucket over all overlapping display bins. This also
    # renders very short recordings without artificial empty stripes.
    maxima = [0.0] * POINTS
    energies = [0.0] * POINTS
    counts = [0.0] * POINTS
    position = 0
    for peak, energy, count in buckets:
        start = position
        end = position + count
        first = min(POINTS - 1, start * POINTS // frames)
        last = min(POINTS - 1, (end * POINTS - 1) // frames)
        for index in range(first, last + 1):
            overlap = max(0, min(end, (index + 1) * frames / POINTS)
                          - max(start, index * frames / POINTS))
            maxima[index] = max(maxima[index], peak)
            energies[index] += energy * overlap / count
            counts[index] += overlap * 2
        position = end
    values = [p * .3 + math.sqrt(e / max(1, n)) * .7
              for p, e, n in zip(maxima, energies, counts)]
    maximum = max(.001, max(values))
    return {"schemaVersion": 1, "durationSeconds": round(frames / SAMPLE_RATE, 4),
            "peaks": [round(max(.012, (v / maximum) ** 1.6), 5) for v in values]}


if __name__ == "__main__":
    print(json.dumps(analyze(sys.argv[1]), separators=(",", ":")))
