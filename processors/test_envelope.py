import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from envelope import analyze


class EnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.previous = os.environ.get("WAVEFORM_FFMPEG")
        self.binary = os.environ.get("WAVEFORM_FFMPEG") or shutil.which("ffmpeg")
        if not self.binary:
            self.skipTest("Set WAVEFORM_FFMPEG to an FFmpeg executable")
        os.environ["WAVEFORM_FFMPEG"] = self.binary

    def tearDown(self):
        self.directory.cleanup()
        if self.previous is None:
            os.environ.pop("WAVEFORM_FFMPEG", None)
        else:
            os.environ["WAVEFORM_FFMPEG"] = self.previous

    def audio(self, name, source="sine=frequency=440:duration=1", args=()):
        path = self.root / name
        subprocess.run([self.binary, "-nostdin", "-v", "error", "-f", "lavfi",
                        "-i", source, *args, str(path)], check=True)
        return path

    def assertEnvelope(self, result):
        self.assertEqual(len(result["peaks"]), 720)
        self.assertTrue(all(math.isfinite(v) and .012 <= v <= 1 for v in result["peaks"]))
        self.assertGreater(result["durationSeconds"], 0)

    def test_common_audio_formats(self):
        for extension in ("wav", "aiff", "flac", "mp3", "m4a"):
            with self.subTest(extension=extension):
                self.assertEnvelope(analyze(self.audio("tone." + extension)))

    def test_large_file_and_original_unchanged(self):
        path = self.audio("large.wav", "sine=frequency=440:sample_rate=192000:duration=48",
                          ("-ac", "2", "-c:a", "pcm_s32le"))
        self.assertGreater(path.stat().st_size, 64 * 1024 * 1024)
        def digest():
            with path.open("rb") as file:
                return hashlib.file_digest(file, "sha256").hexdigest()
        before = digest()
        result = analyze(path)
        self.assertEnvelope(result)
        self.assertAlmostEqual(result["durationSeconds"], 48, places=2)
        self.assertEqual(before, digest())

    def test_opposite_phase_stereo_does_not_cancel(self):
        path = self.audio("stereo.wav", "aevalsrc=0.5*sin(440*2*PI*t)|-0.5*sin(440*2*PI*t):d=1")
        result = analyze(path)
        self.assertGreater(sum(result["peaks"]) / 720, .5)

    def test_silence_and_short_audio(self):
        result = analyze(self.audio("silence.wav", "anullsrc=r=8000:cl=stereo:d=0.05"))
        self.assertEnvelope(result)
        self.assertEqual(set(result["peaks"]), {.012})

    def test_invalid_audio_fails(self):
        path = self.root / "invalid.wav"
        path.write_bytes(b"not an audio file")
        with self.assertRaises(ValueError):
            analyze(path)

    def test_decoder_deadline(self):
        path = self.audio("timeout.wav")
        with self.assertRaises(TimeoutError):
            analyze(path, timeout=.000001)


if __name__ == "__main__":
    unittest.main()
