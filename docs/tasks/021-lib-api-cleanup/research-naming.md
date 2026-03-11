# Research: Naming Review

Reviewing struct, trait, and function names in the public API for clarity.

## 1. `InferenceSession` trait

**Problem**: "Session" is borrowed from ONNX Runtime's C++ API (`Ort::Session`). It implies
lifecycle/state management rather than what it actually does: run a model. Most users won't think
in terms of "sessions".

**Alternatives**:

| Name | Pros | Cons |
|------|------|------|
| `Model` | simple, intuitive | too generic, clashes with domain "model" (the beat model) |
| `ModelRunner` | clear action | a bit long |
| `ModelSession` | explicit | still has "Session" |
| `Backend` | common in ML libs | usually refers to the runtime, not a single loaded model |
| `LoadedModel` | descriptive | verbose |

**Recommendation**: `Model` — it's what users load and run. The generic-ness is fine because it's
a trait, not a concrete type. Combined with the trait bound `BeatThis<M: Model>`, it reads
naturally: "a beat tracker parameterized by its model type."

## 2. `InferenceRuntime` trait

**Problem**: "Runtime" is vague — it could mean the whole library. The trait's only job is to load
models from ONNX files. The name doesn't convey that.

**Alternatives**:

| Name | Pros | Cons |
|------|------|------|
| `ModelLoader` | says exactly what it does | doesn't hint at runtime/backend |
| `Runtime` | shorter | still vague |
| `Backend` | common term | usually means more than just loading |
| `ModelFactory` | pattern name | Java-esque |

**Recommendation**: `ModelLoader` — the trait has a single method `load_model`, so the name
matches perfectly. `OrtRuntime` becomes `OrtLoader`, `RtenRuntime` becomes `RtenLoader`. Or keep
`OrtRuntime`/`RtenRuntime` as the concrete struct names since "runtime" makes more sense for
those (they represent an inference runtime).

**Alternative recommendation**: Drop the prefix entirely. If `InferenceSession` becomes `Model`,
then `InferenceRuntime` could just be `Runtime`:

```rust
pub trait Runtime {
    type Model: Model;
    fn load_model(&self, path: &Path) -> Result<Self::Model>;
}
```

This reads well: `OrtRuntime` implements `Runtime`, produces `OrtModel`s. `RtenRuntime` implements
`Runtime`, produces `RtenModel`s.

## 3. `BeatInference` struct

**Problem**: "Inference" is ML jargon. This struct runs the beat/downbeat prediction model on
mel chunks. The name doesn't tell you it works on mel spectrograms or that it handles chunking.

**Alternatives**:

| Name | Pros | Cons |
|------|------|------|
| `BeatPredictor` | domain-clear | still doesn't mention chunking |
| `BeatModel` | simple | clashes if `Model` is the trait name |
| `BeatDetector` | standard MIR term | could be confused with the whole pipeline |
| `ChunkedPredictor` | describes the mechanism | loses domain context |

**Recommendation**: `BeatPredictor` — it predicts beats from mel spectrograms. If `InferenceSession`
is renamed to `Model`, there's no clash.

Not a blocker since `BeatInference` is proposed to be made private, but cleaner internal naming
still helps maintainability.

## 4. `MelProcessor` struct

**Problem**: "Processor" is generic. This struct computes mel spectrograms from audio using an
ONNX model. It's more like a mel spectrogram extractor/computer.

**Alternatives**:

| Name | Pros | Cons |
|------|------|------|
| `MelExtractor` | clear action | — |
| `MelComputer` | descriptive | sounds like hardware |
| `MelModel` | short | if `Model` is the trait name, reads oddly |
| `MelTransform` | signal-processing term | less intuitive |
| `Spectrogram` | simple | too generic |

**Recommendation**: `MelExtractor` — standard term in audio ML. "Extract mel features" is common
phrasing.

Same caveat as `BeatInference` — proposed to be made private.

## 5. `PostProcessor` struct

**Problem**: Very generic. In beat tracking, the post-processing step is peak picking +
thresholding on logits. The name doesn't convey what kind of post-processing.

**Alternatives**:

| Name | Pros | Cons |
|------|------|------|
| `PeakPicker` | describes algorithm | doesn't mention time conversion |
| `BeatPostProcessor` | domain-scoped | still vague |
| `LogitDecoder` | ML term for logits→events | niche |
| `BeatDecoder` | clear | "decoder" might imply neural decoder |

**Recommendation**: `PeakPicker` or keep `PostProcessor`. Since this is becoming private, it's
lower priority.

## 6. Overloaded `.process()` method

**Problem**: `MelProcessor::process()`, `BeatInference::process()`, and `PostProcessor::process()`
all use the same method name but do completely different things with different signatures. This
is fine for internal code but makes it harder to follow the pipeline.

**Alternatives**:

| Struct | Current | Suggested |
|--------|---------|-----------|
| `MelProcessor` | `process(samples)` | `extract(samples)` or `compute(samples)` |
| `BeatInference` | `process(mel)` | `predict(mel)` or `infer(mel)` |
| `PostProcessor` | `process(logits)` | `pick_peaks(logits)` or `decode(logits)` |

**Recommendation**: Give each a distinct verb. `extract`, `predict`, `decode` would make the
pipeline read as: extract mel → predict beats → decode peaks.

Low priority since all three are proposed to be private.

## 7. `BeatThis` struct field names

**Problem**: The field `post` is abbreviated and unclear. `inference` is ML jargon.

| Current | Suggested |
|---------|-----------|
| `mel` | `mel` (fine) |
| `inference` | `predictor` (if struct is renamed to `BeatPredictor`) |
| `post` | `post_processor` or `decoder` |

Moot if fields become private (as proposed in the API cleanup).

## 8. `Tensor` struct

**Problem**: Not really a problem — `Tensor` is universally understood. But it's very generic
for a public type. Users might confuse it with tensors from other libraries (ndarray, burn, etc).

**Recommendation**: Keep `Tensor`. It's simple and the crate doesn't depend on other tensor
libraries at the API level. Could consider a type alias or newtype if conflicts arise, but not
worth changing now.

## 9. `OrtRuntime` / `RtenRuntime` naming consistency

**Problem**: If the trait is renamed, should the concrete types follow?

| If trait is... | Ort type | Rten type |
|----------------|----------|-----------|
| `Runtime` | `OrtRuntime` (keep) | `RtenRuntime` (keep) |
| `ModelLoader` | `OrtLoader` | `RtenLoader` |
| `Backend` | `OrtBackend` | `RtenBackend` |

**Recommendation**: If trait becomes `Runtime`, keep `OrtRuntime`/`RtenRuntime`. They already
read naturally.

## 10. `load_audio` / `AudioData`

No issues. Clear and standard.

## 11. `calculate_bpm` vs `bpm`

**Problem**: `calculate_bpm` is verbose. In a library context, the calculation is implied.

**Recommendation**: Could shorten to `bpm(analysis)` but `calculate_bpm` is also fine — it's
explicit and only typed once. Low priority.

---

## Summary of recommendations

| Priority | Current | Proposed | Reason |
|----------|---------|----------|--------|
| **High** | `InferenceSession` (trait) | `Model` | Simpler, more intuitive |
| **High** | `InferenceRuntime` (trait) | `Runtime` | Shorter, paired with `Model` |
| Medium | `BeatInference` | `BeatPredictor` | Clearer (becoming private) |
| Medium | `MelProcessor` | `MelExtractor` | Standard term (becoming private) |
| Low | `PostProcessor` | `PeakPicker` | More specific (becoming private) |
| Low | `.process()` everywhere | distinct verbs | Readability (all becoming private) |
| Low | `OrtRuntime` / `RtenRuntime` | keep as-is | Already clear |

The high-priority renames (`InferenceSession` → `Model`, `InferenceRuntime` → `Runtime`)
affect the public API and should be decided before the API cleanup in task 021.
