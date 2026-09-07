# Hymn-based sermon selection

The service-order rule is: select the audio between the final two substantial
**sung hymns**. This replaces the earlier Claude prompt, scripture/prayer section
inference, timestamp-correction heuristics, retries, and duration-based content
combination selector. There is no AI boundary prompt or old-detector fallback.

1. Classify music, singing/choir, and speech from the audio locally.
2. Join music separated by at most five seconds, to keep hymn verses together.
3. Ignore events shorter than 90 seconds, including back-to-back short closing
   responses. Exactly 90 seconds qualifies for consideration.
4. Require sustained music (at least 60% of an event) and singing/choir in at
   least 20% of its detected music. Instrumental introductions can belong to a
   hymn; purely instrumental special music, preludes, and communion music do
   not count as sung hymns.
5. Start at the end of the second-to-last qualifying hymn. Include the spoken
   material after it, including scripture and prayer.
6. Search only the last 120 seconds before the final hymn for an Amen followed
   directly by a hymn announcement within 30 seconds. Also accept Amen as the
   final transcribed word within 30 seconds of the hymn. Keep up to 0.3 seconds
   after the word, without including the announcement. Otherwise, stop at the
   start of the final hymn's music. Without word timestamps, use that fallback.
7. Pause for the existing human review before duration fitting. Broadcast target
   length has no influence on automatic sermon boundaries.

The classifier operates on overlapping 0.975-second windows with a 0.48-second
hop. Its event boundaries are suggestions at that resolution, not guaranteed
sample-exact transitions. The editor still supports exact manual adjustments.
Fewer than two qualifying hymns, model problems, or decoding problems open the
full source for review with a warning; automatic teaser selection is skipped.
In that situation, select the sermon and, for a dynamic broadcast, the teaser
in the editor. The older direct `run_pipeline` entry point reports the failure
instead of rendering a guessed full-service range.

The fixed service parameters are in `config.py`: `HYMN_MIN_DURATION_SECONDS`,
`HYMN_MAX_GAP_SECONDS`, and `HYMN_END_SEARCH_SECONDS`. Changing music classifier
thresholds should be validated against recordings, not used to fit a target
broadcast length.

## Model installation and provenance

Install the requirements, then run as the service account:

```sh
python -m pipeline.music_detector
```

This downloads about 16 MB to `state/models/yamnet.onnx`, or `HYMN_MODEL_PATH` if
configured. Automatic analysis also installs it on first use. The application
verifies the pinned SHA-256 before loading and reuses a CPU inference session.
After installation, music detection needs neither network access nor an API key.
The existing transcription and automatic teaser services retain their own
requirements. Model files live outside job folders and survive job cleanup.

- Upstream: [Google YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet),
  an AudioSet classifier with 521 sound classes.
- ONNX conversion: [audiomagic/yamnet-onnx](https://huggingface.co/audiomagic/yamnet-onnx),
  which includes the waveform preprocessing graph and reports unchanged weights.
- Pinned revision: `f25b741c2f0bdc6d7e6db24b5fddda23347dbafd`.
- Model SHA-256: `d3835ffbbd4a1bb3e777f0ca217b5007907f5171dd5d17c4236b95b2af8f908e`.
- Model license: Apache 2.0, copyright Google LLC. The upstream license is
  [included with the conversion](https://huggingface.co/audiomagic/yamnet-onnx/blob/f25b741c2f0bdc6d7e6db24b5fddda23347dbafd/LICENSE).
  Google's AudioSet class labels are CC BY 4.0. No model weights or service audio
  are committed to this repository.

ONNX Runtime is an explicit dependency; it was already a transitive dependency
of faster-whisper. No TensorFlow installation is required. Audio conversion uses
the application's existing ffmpeg dependency. Inference reads about 31 seconds
at a time, with overlap and final padding so chunk boundaries do not lose audio.

## September 6, 2026 reference

Reference: [Grace Free Lutheran Church service](https://www.youtube.com/watch?v=kbz86frmaYc),
approximately 1 hour 20 minutes. The music pass over the full recording produced:

| Event | Suggested start | Suggested end | Treatment |
|---|---|---|---|
| First sung hymn | 7:46.56 | 10:15.84 | Qualifying hymn |
| Second-to-last sung hymn | 30:48.96 | 32:48.96 | Sermon begins afterward |
| Final sung hymn | 1:01:09.12 | 1:04:04.80 | Outer limit for sermon end |
| Closing musical response | 1:19:04.80 | About 1:20:22 | Below 90 seconds; ignored |

The recording also contains instrumental passages, including after the final
sung hymn. Those do not alter hymn selection. A local faster-whisper `small.en`
transcription of the opening and closing regions places the scripture
introduction after the preceding hymn and the prayer's Amen at
1:00:43.94–1:00:44.08, followed by the hymn announcement at 1:00:48.68.

Result: **32:48.96–1:00:44.38**, a 27:55.42 selection. Without the word transcript,
the expected fallback is **32:48.96–1:01:09.12**. This is one reference recording,
not a reliability claim across all services. Different transcription backends
may supply slightly different word times or require the before-music fallback.

The normal test suite uses synthetic music events and transcript fixtures and
does not download models or call APIs. To run the real-model regression, first
obtain the complete reference audio and install the model, then run:

```sh
HYMN_REFERENCE_AUDIO=/path/to/full-reference.wav python -m pytest tests/test_hymn_reference.py
```

That test reruns audio classification and uses a short fixed word fixture for
the closing Amen/announcement. It does not call or evaluate transcription.
