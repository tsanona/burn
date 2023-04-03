# Text Classification

With this example, you are going to train a transformer text classification model on GPU using the `tch` backend, and run inference on CPU using the `ndarray` backend.

```bash
git clone https://github.com/burn-rs/burn.git
cd burn
# Use the --release flag to really speed up training.
export TORCH_CUDA_VERSION=cu117                                                     # Set the cuda version
cargo run --example text-classification-ag-news --release --features training f16   # Train on the ag news dataset
cargo run --example text-classification-ag-news --release --features inference      # Run inference on the ag news dataset

cargo run --example text-classification-db-pedia --release --features training f16  # Train on the db pedia dataset
cargo run --example text-classification-db-pedia --release --features inference     # Run inference db pedia dataset
```