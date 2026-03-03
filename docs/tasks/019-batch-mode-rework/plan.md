# 019 - Batch Mode Rework: Implementation Plan

## Step 1: Add `glob` dependency

- Add `glob = "0.3"` to `Cargo.toml`

## Step 2: Refactor input resolution (`src/main.rs`)

Extract a `resolve_input` function that unifies file/directory/glob handling:

```rust
enum InputMode {
    SingleFile(PathBuf),
    Batch { files: Vec<PathBuf>, summary_dir: PathBuf },
}

fn resolve_input(input: &Path, recursive: bool) -> anyhow::Result<InputMode>
```

Logic:
1. `input.is_file()` → `SingleFile`
2. `input.is_dir()` → `Batch { files: find_audio_files(..), summary_dir: input }`
3. Pattern contains `*`, `?`, `[` → expand with `glob::glob()`, filter to audio extensions → `Batch { files, summary_dir: cwd }`
4. Otherwise → error

Extract `is_audio_extension(path) -> bool` from the existing `collect_audio_files` logic.

Remove the `cli.input.is_dir()` check in `main()` — use `resolve_input` instead.

## Step 3: Refactor `write_outputs` to accept explicit flags (`src/main.rs`)

Currently `write_outputs` takes `&Cli`. Refactor to take individual flag values so batch mode can override the default:

```rust
struct OutputFlags {
    json: Option<String>,
    beats: Option<String>,
    click: Option<String>,
    mix: Option<String>,
    overwrite: bool,
}
```

- `write_outputs(input, result, flags)` instead of `write_outputs(input, result, cli)`
- `has_output_flags(flags)` checks the struct
- Single-file mode: builds `OutputFlags` directly from cli
- Batch mode: if no flags set, defaults to `OutputFlags { json: Some(""), .. }` (auto-named JSON)

## Step 4: Rework batch summary structs (`src/output.rs`)

Replace old structs:

```rust
// OLD — remove
pub struct BatchFileOutput { file, json: JsonOutput, duration_secs, processing_time_secs }
pub struct BatchOutput { files: Vec<BatchFileOutput>, summary: BatchSummary }

// NEW
pub struct BatchFileEntry {
    pub input: String,
    pub duration_secs: f32,
    pub processing_time_secs: f32,
    pub outputs: Vec<String>,
}

pub struct BatchSummaryOutput {
    pub files: Vec<BatchFileEntry>,
    pub summary: BatchSummary,
}
```

Update `write_batch_json` signature to take `&BatchSummaryOutput`.

`BatchSummary` stays the same.

## Step 5: Rework `run_batch` (`src/main.rs`)

Rewrite `run_batch` to:
1. Accept `files: Vec<PathBuf>` and `summary_dir: &Path` instead of `dir: &Path`
2. Determine effective output flags (default to `--json` if none set)
3. For each file: process → `write_outputs` → collect `BatchFileEntry`
4. Always write `beat_this.json` to `summary_dir` using `BatchSummaryOutput`

Remove the old conditional logic (`if !has_output_flags` / `else`).

## Step 6: Update `main()` (`src/main.rs`)

Replace:
```rust
let is_batch = cli.input.is_dir();
// ... ensure cli.input.exists() ...
// ... if is_batch { run_batch } else { run_pipeline }
```

With:
```rust
match resolve_input(&cli.input, cli.recursive)? {
    InputMode::SingleFile(path) => run_pipeline(..),
    InputMode::Batch { files, summary_dir } => run_batch(.., files, summary_dir),
}
```

Move the `cli.input.exists()` validation into `resolve_input` (for file/dir modes).

## Step 7: Update tests

### `src/output.rs`
- Update `test_write_batch_json` to use new `BatchFileEntry` / `BatchSummaryOutput` structs
- Verify the new JSON shape: `files[].input`, `files[].outputs`, no `files[].beats`

### `scripts/integration-test.py`
- Update batch test to verify new `beat_this.json` format (summary only, no beat data)
- Verify per-file `.json` files are created by default
- Add a glob pattern test if feasible

## Step 8: Update README.md

- Document the new batch default (per-file JSON)
- Document glob pattern support with quoting
- Update the CLI options table
- Update examples

## File Change Summary

| File | Changes |
|------|---------|
| `Cargo.toml` | Add `glob = "0.3"` |
| `src/main.rs` | Add `resolve_input`, `InputMode`, `OutputFlags`, `is_audio_extension`. Refactor `run_batch`, `write_outputs`, `main`. Remove old `is_batch` logic. |
| `src/output.rs` | Replace `BatchFileOutput` → `BatchFileEntry`, `BatchOutput` → `BatchSummaryOutput`. Update `write_batch_json`. |
| `scripts/integration-test.py` | Update batch test assertions |
| `README.md` | Update docs |
