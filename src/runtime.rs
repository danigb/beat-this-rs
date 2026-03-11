pub mod ort;

pub mod rten;

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

/// Simple f32 tensor with shape (row-major / C-order).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// A loaded model ready for inference.
pub trait Model {
    /// Run inference with named inputs, return named outputs.
    fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<HashMap<String, Tensor>>;
}

/// Factory for loading models from ONNX files.
pub trait Runtime {
    type Model: Model;

    fn load_model(&self, path: &Path) -> Result<Self::Model>;
}
