use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use ndarray::ArrayD;

/// Convert an `ort::Error<R>` into `anyhow::Error`.
///
/// As of ort 2.0.0-rc.12, `ort::Error` is generic over the builder/recovery
/// type `R`, which is not `Send + Sync`, so it can no longer be converted into
/// `anyhow::Error` via `?`. We flatten it to its `Display` string instead.
fn ort_err<R>(e: ort::Error<R>) -> anyhow::Error {
    anyhow::anyhow!(e.to_string())
}
use ort::environment::Environment;
use ort::ep::{CoreML, ExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{DynValue, Value};

use super::{Model, Runtime, Tensor};

/// Ort-based ONNX inference runtime.
///
/// Automatically tries CoreML on macOS (falls back to CPU if unavailable).
///
/// ONNX Runtime allows one environment per process. If the application has
/// already created one with `ort::init()`, pass it in `environment`;
/// otherwise the first `load_model` creates one and later runtimes share it.
pub struct OrtRuntime {
    pub optimization_level: GraphOptimizationLevel,
    pub intra_threads: usize,
    pub profiling_path: Option<PathBuf>,
    pub environment: Option<Environment>,
}

impl Default for OrtRuntime {
    fn default() -> Self {
        Self {
            optimization_level: GraphOptimizationLevel::Level3,
            // 0 = let ORT pick the thread count automatically.
            // This is critical for performance: on Apple Silicon M1, intra_threads=1
            // gives ~15.7s for beat inference on a 4.5-min track, while auto (0) gives
            // ~3.4s — a 4.6x speedup. The bottleneck is batched MatMul in the attention
            // layers (73% of inference time), which parallelizes well across cores.
            intra_threads: 0,
            profiling_path: None,
            environment: None,
        }
    }
}

/// The environment created by this crate when the application did not pass
/// one, shared by every `OrtRuntime` (ort 2.0.0-rc.14 requires one to build a
/// session, and ONNX Runtime allows only one per process).
fn environment() -> Environment {
    static ENV: std::sync::OnceLock<Environment> = std::sync::OnceLock::new();
    ENV.get_or_init(|| {
        ort::init().with_name("beat-this").build().expect(
            "initialising ONNX Runtime (is libonnxruntime installed, or ORT_DYLIB_PATH set?)",
        )
    })
    .clone()
}

impl OrtRuntime {
    /// Check if CoreML is available in the loaded ORT runtime.
    pub fn is_coreml_available(&self) -> bool {
        CoreML::is_available().unwrap_or(false)
    }
}

impl Runtime for OrtRuntime {
    type Model = OrtModel;

    fn load_model(&self, path: &Path) -> Result<OrtModel> {
        // `GraphOptimizationLevel` is `Copy` as of ort 2.0.0-rc.14.
        let optimization_level = self.optimization_level;
        let environment = self.environment.clone().unwrap_or_else(environment);
        let mut builder = Session::builder(&environment)
            .map_err(ort_err)?
            .with_optimization_level(optimization_level)
            .map_err(ort_err)?
            .with_intra_threads(self.intra_threads)
            .map_err(ort_err)?
            .with_execution_providers([CoreML::default().build()])
            .map_err(ort_err)?;
        if let Some(ref profile_path) = self.profiling_path {
            builder = builder.with_profiling(profile_path).map_err(ort_err)?;
        }
        let session = builder.commit_from_file(path).map_err(ort_err)?;
        Ok(OrtModel { session })
    }
}

/// An ort-backed model wrapping `ort::Session`.
pub struct OrtModel {
    session: Session,
}

impl OrtModel {
    /// End profiling and flush the trace JSON file. Returns the profile file path.
    pub fn end_profiling(&mut self) -> Result<String> {
        self.session.end_profiling().map_err(ort_err)
    }
}

impl Model for OrtModel {
    fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<HashMap<String, Tensor>> {
        // Convert Tensor inputs to ort DynValues
        let ort_inputs: Vec<(String, DynValue)> = inputs
            .iter()
            .map(|(name, tensor)| {
                let shape: Vec<usize> = tensor.shape.clone();
                // The ort backend is a maintainer oracle; this input copy is accepted here.
                let array = ArrayD::from_shape_vec(shape, tensor.data.clone())?;
                let value: DynValue = Value::from_array(array).map_err(ort_err)?.into_dyn();
                Ok((name.to_string(), value))
            })
            .collect::<Result<Vec<_>>>()?;

        // Build input refs for session.run()
        let input_refs: Vec<(&str, &DynValue)> = ort_inputs
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect();

        let outputs = self.session.run(input_refs).map_err(ort_err)?;

        // Convert outputs to Tensor map
        let mut result = HashMap::new();
        for (name, value) in outputs.iter() {
            let (shape, data) = value.try_extract_tensor::<f32>().map_err(ort_err)?;
            let tensor = Tensor {
                shape: shape.iter().map(|&d| d as usize).collect(),
                data: data.to_vec(),
            };
            result.insert(name.to_string(), tensor);
        }

        Ok(result)
    }
}
