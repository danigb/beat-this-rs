use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use rten::{Model, NodeId};
use rten::Value as RtenValue;
use rten_tensor::{AsView, Layout};

use super::{InferenceRuntime, InferenceSession, Tensor};

/// Pure-Rust ONNX inference runtime backed by rten.
///
/// Uses Rayon internally for multi-threaded inference (defaults to physical
/// core count). No configuration needed — just load and run.
pub struct RtenRuntime;

impl InferenceRuntime for RtenRuntime {
    type Session = RtenSession;

    fn load_model(&self, path: &Path) -> Result<RtenSession> {
        let model = Model::load_file(path)?;

        // Build name→NodeId map for inputs
        let input_map: HashMap<String, NodeId> = model
            .input_ids()
            .iter()
            .filter_map(|&id| {
                let info = model.node_info(id)?;
                let name = info.name()?;
                Some((name.to_string(), id))
            })
            .collect();

        // Build NodeId→name map for outputs (reverse lookup when returning results)
        let output_names: Vec<(NodeId, String)> = model
            .output_ids()
            .iter()
            .filter_map(|&id| {
                let info = model.node_info(id)?;
                let name = info.name()?;
                Some((id, name.to_string()))
            })
            .collect();

        let output_ids: Vec<NodeId> = model.output_ids().to_vec();

        Ok(RtenSession {
            model,
            input_map,
            output_names,
            output_ids,
        })
    }
}

pub struct RtenSession {
    model: Model,
    /// "mel_spectrogram" → NodeId
    input_map: HashMap<String, NodeId>,
    /// [(NodeId, "beat"), (NodeId, "downbeat")]
    output_names: Vec<(NodeId, String)>,
    /// Ordered output node IDs for model.run()
    output_ids: Vec<NodeId>,
}

impl InferenceSession for RtenSession {
    fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<HashMap<String, Tensor>> {
        // Convert named inputs to (NodeId, Value) pairs
        let rten_inputs: Vec<(NodeId, RtenValue)> = inputs
            .iter()
            .map(|(name, tensor)| {
                let node_id = self
                    .input_map
                    .get(*name)
                    .ok_or_else(|| anyhow!("rten: unknown input name '{}'", name))?;
                let value = RtenValue::from_shape(tensor.shape.as_slice(), tensor.data.clone())
                    .map_err(|e| anyhow!("rten: failed to create input tensor '{}': {}", name, e))?;
                Ok((*node_id, value))
            })
            .collect::<Result<Vec<_>>>()?;

        // model.run takes Vec<(NodeId, ValueOrView)> — convert via (&val).into()
        let inputs_with_views: Vec<_> = rten_inputs
            .iter()
            .map(|(id, val)| (*id, val.into()))
            .collect();

        let outputs = self.model.run(inputs_with_views, &self.output_ids, None)?;

        // Convert outputs to named Tensor map
        let mut result = HashMap::new();
        for (&id, value) in self.output_ids.iter().zip(outputs.into_iter()) {
            let name = self
                .output_names
                .iter()
                .find(|(nid, _)| *nid == id)
                .map(|(_, n)| n.clone())
                .unwrap_or_else(|| format!("output_{:?}", id));

            let rten_tensor = value
                .into_tensor::<f32>()
                .ok_or_else(|| anyhow!("rten: output '{}' is not f32", name))?;
            let shape: Vec<usize> = rten_tensor.shape().to_vec();
            let data: Vec<f32> = rten_tensor.to_vec();

            result.insert(name, Tensor { shape, data });
        }

        Ok(result)
    }
}
