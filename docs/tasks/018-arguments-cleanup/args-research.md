# Output Arguments Research

## Current state

### Single-file mode (`beat-this input.wav`)

| Arg | Type | Behavior |
|-----|------|----------|
| (default) | — | Prints JSON to stdout |
| `--output-beats` | bool | Prints plain text to stdout *instead* of JSON |
| `--output-click <PATH>` | required path | Writes click WAV to path |
| `--output-mixed <PATH>` | required path | Writes mixed WAV to path |

Inconsistencies:
- `--output-beats` is a bool that changes stdout format
- `--output-click` and `--output-mixed` require an explicit path
- No way to write JSON to a file (only stdout)
- No skip-if-exists behavior

### Batch mode (`beat-this ./dir/`)

- Always writes `beat-this.json` in the input directory
- `--output-beats` writes `.beats` files alongside each input file
- `--output-click` and `--output-mixed` are not wired up in batch mode

## Proposed design

Four unified `--output-*` flags, each with an optional path:

```
--output-json [FILE]      Write JSON          (default ext: .json)
--output-beats [FILE]     Write beats file    (default ext: .beats)
--output-click [FILE]     Write click WAV     (default ext: .click.wav)
--output-mix [FILE]       Write mixed WAV     (default ext: .mix.wav)
```

### Rules

1. **No `--output-*` flags**: print JSON to stdout (current default, machine-friendly)
2. **Any `--output-*` flag present**: write files, print a summary to stderr, nothing to stdout
3. **Optional path**: if given, use it; if omitted, derive from input filename (`song.wav` → `song.json`, etc.)
4. **`--overwrite`**: global flag. Without it, skip writing if file exists (log skip to stderr)
5. **Batch mode**: file path argument is ignored; output is always derived from each input filename. Summary includes written/skipped files per input

### Single-file examples

```bash
# Default: JSON to stdout (pipe-friendly)
beat-this input.wav

# Write JSON + beats files alongside input
beat-this input.wav --output-json --output-beats
# → writes input.json, input.beats
# → stderr summary: "Wrote input.json, input.beats"

# Explicit paths
beat-this input.wav --output-json results.json --output-click click.wav

# All four outputs, auto-named
beat-this input.wav --output-json --output-beats --output-click --output-mix

# Overwrite existing
beat-this input.wav --output-json --overwrite
```

### Batch examples

```bash
# Default: writes beat-this.json in dir (current behavior)
beat-this ./music/ -r

# Write per-file JSON + beats
beat-this ./music/ -r --output-json --output-beats
# → song1.json, song1.beats, song2.json, song2.beats, ...
```

## Suggestion: drop `--output-` prefix

The `--output-` prefix is verbose. Since these are the main action of the tool, shorter names are more ergonomic:

```
--json [FILE]
--beats [FILE]
--click [FILE]
--mix [FILE]
```

Examples become cleaner:

```bash
beat-this input.wav --json --beats
beat-this input.wav --json results.json --click
beat-this ./music/ -r --json --beats --click
```

The trade-off: `--json` and `--beats` are less self-documenting than `--output-json`. But in context of a CLI whose entire purpose is producing beat output, the meaning is clear. Tools like `ffmpeg` and `sox` use short output flags for the same reason.

This also avoids the awkward `--output-mix` vs `--output-mixed` naming question.

## Implementation notes

- clap supports optional values with `num_args = 0..=1` and `default_missing_value`
- The "derive from input" logic: `input.with_extension("json")` etc.
- Skip-if-exists: `path.exists() && !cli.overwrite` → log skip, continue
- Batch summary struct needs a `written_files: Vec<String>` per file entry
- `print_json_stdout` can be removed once `--output-json` replaces it (or kept as the no-flags default)
