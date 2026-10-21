# Submission serialization constraint

User's Pyodide traceback fails inside json.dumps, not inside machine execution.
`python3.12 -m codex.profile_submission_json` reports:

|Artifact|Packets|Default instruction-list JSON bytes|
|---|---:|---:|
|983|9504|4976716|
|979|4064537|2202463376|

983 was actually serialized locally.979 was calculated exactly from the
lossless recipes: each varying nibble source occurs4096 times per65536-entry
block; JSON punctuation and original operand digit lengths are included.
The formula is independently checked on an expanded16-entry example. No
2.2GB string was allocated for the large candidate.

The2.20GB figure is only the instruction-list JSON, not peak heap consumption
or any surrounding website response. Python instruction objects, JSON chunks
and final string can coexist. Compression of source text and shared Python
tuples do not eliminate repeated values in ordinary JSON output.

Treat979 as a locally verified performance result that failed actual browser
submission, not a working web submission.983 is a much smaller fallback but
its browser acceptance has not been independently confirmed. Future sub900
designs must avoid multi-million-entry expanded tables and profile default
JSON size before proposing a browser upload. The website's precise memory
limit remains unknown; do not invent a hard threshold or alter its harness.

## Current portable submission (2026-10-21)

The root `perf_takehome.py` now contains the locally generated
865-cycle kernel: 865 packets, 153007 bytes of Python source, and 489061 bytes
of default instruction-list JSON. It passes all 9 official tests and 270
deterministic verification cases on the unchanged frozen simulator. The 270
cases also pass after JSON transport converts instruction tuples to lists.

This is the recommended local artifact. Its standalone builder needs no solver,
external table, predecessor source or input data. Its compressed payload is the
fixed ISA instruction stream, not a lookup table of answers or runtime inputs.
All tree preprocessing and hash work runs in the frozen machine. The readable
compiler, register allocator and verified schedule are preserved in `codex/`.
Browser acceptance itself remains untested. See
`SUB864_865_RESULT_20261021.md` for provenance and exact reproduction.

The exact 865-cycle source is also saved separately as
`perf_takehome_865.py`. Its bytes match the root submission, with SHA256
`58ce2efca049a18ef343ba82ceaeb72fd74f638bd7c8bbc23a14e6e796d72f44`.

The 866 result is preserved in `perf_takehome_866.py`: 866 packets,
153311 source bytes and 496232 JSON bytes.

The 868 result is preserved in `perf_takehome_868.py`: 868 packets,
151945 source bytes and 486761 JSON bytes.

The previous public-kernel reproduction is preserved as `perf_takehome_880.py`:
880 packets, 504890 source bytes and 541539 JSON bytes.

The previous 981-cycle version (9502 packets, 135068 source bytes, 4976726 JSON
bytes) is preserved as `perf_takehome_981_portable.py`. The earlier 979-cycle
memory warning still applies to that separate artifact.
