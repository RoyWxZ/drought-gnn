# DroughtGNN
Graph-based deep learning model for drought prediction

Requires a CUDA 12.x-compatible GPU, as this uses a CUDA-enabled PyTorch package

Datasets and model files are stored in an S3 cloud bucket via Data Version Control (dvc) rather than directly on GitHub because of their large size. The S3 remote allows public read-only downloads through DVC, but write access requires authenticated IAM keys.
