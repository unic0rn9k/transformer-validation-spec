# Transformer validation spec
Currently the spec just trains a single attention head to indentify a missing element in a sequence of integers.

The code is located in [src/main.rs](src/main.rs).
The code uses the [tch-rs](https://crates.io/crates/tch) Torch bindings for optimizing the model.
