"""Local file -> portable waveform and musical-analysis JSON, with no hosting."""
import argparse
import json
from pathlib import Path
import sys

from envelope import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, help="Write JSON here; defaults to standard output")
    parser.add_argument("--waveform-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() == args.audio.resolve():
            raise ValueError("Output cannot replace the original audio")
        result = {"schemaVersion": 1, "waveform": analyze(args.audio)}
        if not args.waveform_only:
            from musical import analyze_music
            result["music"] = analyze_music(args.audio)
        body = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
        if args.output:
            # Exclusive creation also protects pre-existing metadata and links.
            with args.output.open("x") as output:
                output.write(body)
        else:
            print(body, end="")
    except (ValueError, TimeoutError, OSError) as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
