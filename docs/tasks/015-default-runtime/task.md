Given performance results (on my machine only for now), I want two make rtes runtime the default one.

Also I want to explore the possibility of supporting several runtimes at runtime (not compilation). It should use rtes by default and use onnx (dynamic only, I don't want to compile inside the rust binary) with a CLI flag (or similar). Is it possible? Pros/cons.
