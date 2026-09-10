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
