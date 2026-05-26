# DroughtGNN
Graph-based deep learning model for drought prediction

Requires a CUDA 12.x-compatible GPU, as this uses a CUDA-enabled PyTorch package

Datasets and model files stored in S3 cloud bucket via dvc rather than directly on github due to large size (S3 remote for dvc allows public read-only downloads; writes require authenticated IAM keys)