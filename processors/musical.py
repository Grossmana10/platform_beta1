"""Streaming spectral features and conservative, heuristic key/tempo estimates.

These are suggestions, not calibrated probabilities or ground-truth metadata.
No model, network call, or paid service is required.
"""
from array import array
from pathlib import Path
import subprocess
import threading

import numpy as np
from scipy.signal import fftconvolve, find_peaks

from envelope import MAX_FILE_BYTES, ffmpeg_binary

RATE, FFT, HOP = 22050, 4096, 512
NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
PROFILES = {
    "major": np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
    "minor": np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
}


def key_estimate(chroma):
    empty = {"name": None, "confidence": "low", "score": 0.0, "alternatives": []}
    if float(chroma.sum()) < 1e-8:
        return empty
    vector = chroma / chroma.sum()
    # Broad/noisy spectra and single isolated pitches don't establish a key.
    if np.std(vector) < .018 or np.count_nonzero(vector > vector.max() * .18) < 3:
        return empty
    ranked = []
    for mode, profile in PROFILES.items():
        for tonic in range(12):
            correlation = float(np.corrcoef(vector, np.roll(profile, tonic))[0, 1])
            ranked.append((correlation, f"{NAMES[tonic]} {mode}"))
    ranked.sort(reverse=True)
    best, name = ranked[0]
    margin = best - ranked[1][0]
    if best < .5:
        return empty
    confidence = "high" if best > .8 and margin > .14 else "medium" if margin > .06 else "low"
    return {"name": name, "confidence": confidence, "score": round(best, 3),
            "alternatives": [{"name": label, "score": round(score, 3)} for score, label in ranked[1:3]]}


def tempo_estimate(onsets):
    empty = {"bpm": None, "confidence": "low", "score": 0.0, "alternatives": []}
    values = np.asarray(onsets, dtype=np.float64)
    fps = RATE / HOP
    if len(values) < fps * 6 or values.max(initial=0) < 1e-8:
        return empty
    # FFT-window fluctuations in a sustained tone aren't rhythmic attacks.
    values[values < values.max() * .015] = 0
    # Remove tiny background fluctuations; cap extreme isolated transients.
    values = np.maximum(0, values - np.percentile(values, 60))
    nonzero = values[values > 0]
    if not len(nonzero):
        return empty
    values = np.minimum(values, np.percentile(nonzero, 95))
    events, _ = find_peaks(values, distance=max(1, int(fps * .12)), prominence=values.max() * .15)
    if len(events) < 5:
        return empty
    centered = values - values.mean()
    ac = fftconvolve(centered, centered[::-1], mode="full")[len(values) - 1:]
    if ac[0] <= 0:
        return empty
    ac /= ac[0]
    low, high = int(fps * 60 / 220), min(len(ac) - 2, int(fps * 60 / 40))
    peaks, _ = find_peaks(ac[low:high + 1])
    candidates = []
    for lag in peaks + low:
        score = float(ac[lag])
        if score < .15:
            continue
        denom = ac[lag - 1] - 2 * ac[lag] + ac[lag + 1]
        offset = .5 * (ac[lag - 1] - ac[lag + 1]) / denom if abs(denom) > 1e-9 else 0
        bpm = 60 * fps / (lag + float(np.clip(offset, -.5, .5)))
        if 40 <= bpm <= 220:
            # Weak preference for conventional beat rates resolves exact
            # metrical ties; half/double time remain explicit alternatives.
            rank = score * np.exp(-.08 * np.log2(bpm / 110) ** 2)
            candidates.append((rank, bpm, score))
    if not candidates:
        return empty
    candidates.sort(reverse=True)
    _, bpm, score = candidates[0]
    alternatives = []
    for value, relation in ((bpm / 2, "half-time"), (bpm * 2, "double-time")):
        if 30 <= value <= 300:
            alternatives.append({"bpm": round(value, 1), "relation": relation})
    for _, value, _ in candidates[1:]:
        if abs(value - bpm) > 3 and all(abs(value - a["bpm"]) > 3 for a in alternatives):
            alternatives.append({"bpm": round(value, 1), "relation": "other candidate"})
            break
    confidence = "high" if score > .65 and len(events) >= 16 else "medium" if score > .35 else "low"
    return {"bpm": round(bpm, 1), "confidence": confidence,
            "score": round(min(1, score), 3), "alternatives": alternatives}


def analyze_music(path, timeout=300):
    path = Path(path).resolve(strict=True)
    if not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("Audio must fit the existing 500 MiB upload limit")
    window = np.hanning(FFT).astype(np.float32)
    frequencies = np.fft.rfftfreq(FFT, 1 / RATE)
    usable = np.flatnonzero((frequencies >= 65) & (frequencies <= 4000))
    chroma = np.zeros(12)
    previous = np.zeros(FFT // 2 + 1)
    onsets = array("f")
    pending = np.empty((0, 2), dtype=np.float32)
    frame_count = 0
    tonal_frames = 0
    expired = threading.Event()
    command = [ffmpeg_binary(), "-nostdin", "-v", "error", "-xerror", "-threads", "1",
               "-protocol_whitelist", "file,pipe", "-i", str(path), "-map", "0:a:0",
               "-vn", "-sn", "-dn", "-ac", "2", "-ar", str(RATE), "-f", "f32le", "pipe:1"]
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
                length = len(chunk) - len(chunk) % 8
                remainder = chunk[length:]
                samples = np.frombuffer(chunk[:length], dtype="<f4").reshape(-1, 2)
                if not np.isfinite(samples).all():
                    raise ValueError("Audio contains non-finite samples")
                frame_count += len(samples)
                pending = np.concatenate((pending, samples))
                start = 0
                while start + FFT <= len(pending):
                    block = pending[start:start + FFT]
                    # Combine channel magnitudes, never opposite-phase samples.
                    spectrum = np.abs(np.fft.rfft(block * window[:, None], axis=0)).mean(axis=1)
                    flux = np.maximum(0, np.log1p(spectrum) - np.log1p(previous))
                    onsets.append(float(flux[usable].mean()))
                    previous = spectrum
                    # Local spectral peaks reduce leakage between pitch classes.
                    peaks, _ = find_peaks(spectrum, prominence=max(1e-7, float(spectrum.max()) * .035))
                    peaks = peaks[(peaks >= usable[0]) & (peaks <= usable[-1])]
                    if len(peaks):
                        logspec = np.log(np.maximum(spectrum, 1e-12))
                        denominator = logspec[peaks - 1] - 2 * logspec[peaks] + logspec[peaks + 1]
                        offset = np.divide(.5 * (logspec[peaks - 1] - logspec[peaks + 1]), denominator,
                                           out=np.zeros_like(denominator), where=np.abs(denominator) > 1e-10)
                        pitches = 69 + 12 * np.log2((peaks + np.clip(offset, -.5, .5)) * RATE / FFT / 440)
                        classes = np.rint(pitches).astype(int) % 12
                        weights = spectrum[peaks] ** .7
                        frame_chroma = np.bincount(classes, weights=weights, minlength=12)
                        chroma += frame_chroma / max(1e-12, frame_chroma.sum())
                        tonal_frames += 1
                    start += HOP
                    # Bound feature storage for exceptionally long low-rate
                    # files. Typical songs are analyzed in their entirety.
                    if len(onsets) >= 2_000_000:
                        raise ValueError("Recording exceeds the analysis duration budget")
                pending = pending[start:].copy()
            code = process.wait()
            if expired.is_set():
                raise TimeoutError("Music analysis exceeded its processing deadline")
            if code or remainder or not frame_count:
                raise ValueError("Audio could not be decoded")
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
                process.wait()
    return {"key": key_estimate(chroma), "tempo": tempo_estimate(onsets),
            "analyzedSeconds": round(frame_count / RATE, 3),
            "method": "spectral-chroma-and-onset-autocorrelation-v1",
            "confidenceMeaning": "Heuristic signal strength, not a probability of correctness",
            "tonalFrames": tonal_frames}
