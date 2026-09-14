import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

from musical import analyze_music


class MusicalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, samples):
        path = self.root / "test.wav"
        wavfile.write(path, 22050, np.asarray(samples, dtype=np.float32))
        return path

    def clicks(self, bpm, duration=24):
        samples = np.zeros(22050 * duration)
        click = np.random.default_rng(12).normal(0, .3, 700) * np.exp(-np.arange(700) / 120)
        for position in np.arange(.5, duration - .1, 60 / bpm):
            start = round(position * 22050)
            samples[start:start + len(click)] += click
        return samples

    def progression(self, minor=False, transpose=0):
        # Tonic, subdominant, dominant, tonic; a pure-tone fixture with known
        # harmony. This checks mechanics, not accuracy on commercial masters.
        chords = ([48, 51, 55], [53, 56, 60], [55, 59, 62], [48, 51, 55]) if minor else (
            [48, 52, 55], [53, 57, 60], [55, 59, 62], [48, 52, 55])
        output = []
        for index, chord in enumerate(chords):
            duration = 5 if index in (0, 3) else 2
            time = np.arange(22050 * duration) / 22050
            section = sum(.12 * np.sin(2 * np.pi * 440 * 2 ** ((note + transpose - 69) / 12) * time)
                          for note in chord)
            section[:1000] *= np.linspace(0, 1, 1000)
            section[-1000:] *= np.linspace(1, 0, 1000)
            output.append(section)
        return np.concatenate(output)

    def test_known_tempos(self):
        for bpm in (80, 120, 150):
            with self.subTest(bpm=bpm):
                result = analyze_music(self.write(self.clicks(bpm)))
                candidates = [result["tempo"]["bpm"]] + [v["bpm"] for v in result["tempo"]["alternatives"]]
                self.assertTrue(any(v and abs(v - bpm) < 2 for v in candidates), result)

    def test_major_minor_and_transposition(self):
        for minor, transpose, expected in ((False, 0, "C major"), (True, 0, "C minor"), (False, 2, "D major")):
            with self.subTest(key=expected):
                result = analyze_music(self.write(self.progression(minor, transpose)))
                self.assertEqual(result["key"]["name"], expected, result)

    def test_silence_is_unknown(self):
        result = analyze_music(self.write(np.zeros(22050 * 8)))
        self.assertIsNone(result["key"]["name"])
        self.assertIsNone(result["tempo"]["bpm"])

    def test_single_note_does_not_establish_key(self):
        time = np.arange(22050 * 8) / 22050
        result = analyze_music(self.write(.3 * np.sin(2 * np.pi * 440 * time)))
        self.assertIsNone(result["key"]["name"])
        self.assertIsNone(result["tempo"]["bpm"])

    def test_antiphase_preserves_key(self):
        samples = self.progression()
        result = analyze_music(self.write(np.column_stack((samples, -samples))))
        self.assertEqual(result["key"]["name"], "C major")

    def test_cli_json_and_overwrite_protection(self):
        path = self.write(self.clicks(120, duration=8))
        command = [sys.executable, str(Path(__file__).with_name("analyze.py")), str(path)]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout)
        self.assertEqual(len(result["waveform"]["peaks"]), 720)
        self.assertIn("tempo", result["music"])
        before = path.stat().st_size
        rejected = subprocess.run(command + ["--output", str(path)], capture_output=True)
        self.assertEqual(rejected.returncode, 1)
        self.assertEqual(path.stat().st_size, before)


if __name__ == "__main__":
    unittest.main()
