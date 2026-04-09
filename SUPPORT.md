# Support Policy

This repository explicitly supports:

- CPython 3.9, 3.11, and 3.13
- Linux, macOS, and Windows

Current CI coverage:

| Runner | Python versions | Coverage level |
| --- | --- | --- |
| `ubuntu-latest` | 3.9, 3.11, 3.13 | Full test suite |
| `macos-latest` | 3.11 | Smoke test |
| `windows-latest` | 3.11 | Smoke test |

Practical notes:

- Video audio extraction requires the `ffmpeg` system binary.
- Some document and tabular formats require optional parser dependencies at runtime.

For the fuller user-facing matrix and caveats, see [`docs/support.md`](docs/support.md).
