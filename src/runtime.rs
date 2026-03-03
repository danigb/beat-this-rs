pub mod ort;

pub mod rten;

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

/// Simple f32 tensor with shape (row-major / C-order).
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// A loaded model session ready for inference.
pub trait InferenceSession {
    /// Run inference with named inputs, return named outputs.
    fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<HashMap<String, Tensor>>;
}

/// Factory for creating sessions from ONNX model files.
pub trait InferenceRuntime {
    type Session: InferenceSession;

    fn load_model(&self, path: &Path) -> Result<Self::Session>;
}
