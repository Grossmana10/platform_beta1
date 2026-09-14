# Platform audio processors

Standalone waveform, estimated musical key, and tempo analysis. No audio files, user data, private Site source, or credentials are included.

## Local use

Requires Python 3.11 or newer.

```sh
pip install -r processors/requirements.txt
python processors/analyze.py /path/to/song.wav --output song.analysis.json
```

The output contains 720 waveform amplitude values, estimated key and BPM, heuristic confidence labels, and alternate tempo readings. Estimates need validation on representative real mixes. Original audio is unchanged.

## Service

Build: `pip install -r processors/requirements.txt`

Start: `python processors/worker.py`

Health: `/health`. A response of `configuration_required` means the service is running without credentials and cannot access or process private audio. `running` means the queue consumer is active, not that end-to-end connectivity has been verified.

Runtime-only settings: `PLATFORM_ORIGIN`, `WAVEFORM_WORKER_SECRET`, and `SITES_DISPATCH_TOKEN` for a gated Site. Never commit their values. The Site must implement the lease-based queue endpoints before enabling the consumer. Persistent queue/results belong to the Site; temporary downloaded audio is deleted after processing.

No paid service is required for local use. Free hosting is suitable for testing and has provider usage and sleep limits.
