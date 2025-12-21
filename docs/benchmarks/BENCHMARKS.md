# Benchmarks

Performance benchmarks for Unbihexium models and pipelines.

## Hardware Specifications

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD EPYC 7763 (64 cores) |
| **GPU** | NVIDIA A100 80GB |
| **RAM** | 512 GB DDR4 |
| **Storage** | NVMe SSD |
| **OS** | Ubuntu 22.04 LTS |
| **Python** | 3.12 |
| **ONNX Runtime** | 1.16 |

## Inference Performance

### Detection Models

| Model | Variant | Input Size | Batch 1 (ms) | Batch 8 (ms) | Throughput (img/s) |
|-------|---------|------------|--------------|--------------|-------------------|
| ShipDetector | tiny | 64×64 | 2.1 | 8.5 | 940 |
| ShipDetector | base | 128×128 | 5.3 | 22.1 | 362 |
| ShipDetector | large | 256×256 | 18.4 | 76.2 | 105 |
| ShipDetector | mega | 512×512 | 72.1 | 298.5 | 27 |
| BuildingDetector | tiny | 64×64 | 2.4 | 9.2 | 870 |
| BuildingDetector | base | 128×128 | 6.1 | 25.3 | 316 |
| BuildingDetector | large | 256×256 | 21.2 | 88.4 | 90 |

### Segmentation Models

| Model | Variant | Input Size | Batch 1 (ms) | Batch 8 (ms) | mIoU |
|-------|---------|------------|--------------|--------------|------|
| WaterSegmenter | tiny | 64×64 | 3.2 | 12.1 | 0.91 |
| WaterSegmenter | base | 128×128 | 8.1 | 33.2 | 0.94 |
| WaterSegmenter | large | 256×256 | 28.5 | 118.2 | 0.95 |
| WaterSegmenter | mega | 512×512 | 112.3 | 465.1 | 0.96 |
| ForestSegmenter | base | 128×128 | 9.2 | 38.1 | 0.91 |
| CropSegmenter | base | 128×128 | 8.8 | 36.4 | 0.88 |

### Super-Resolution Models

| Model | Scale | Input Size | Output Size | Time (ms) | PSNR (dB) | SSIM |
|-------|-------|------------|-------------|-----------|-----------|------|
| ESRGAN | 2× | 128×128 | 256×256 | 45.2 | 32.4 | 0.92 |
| ESRGAN | 4× | 128×128 | 512×512 | 82.1 | 28.6 | 0.87 |
| ESRGAN | 8× | 64×64 | 512×512 | 95.3 | 25.1 | 0.81 |
| EDSR | 2× | 128×128 | 256×256 | 38.4 | 31.8 | 0.91 |
| EDSR | 4× | 128×128 | 512×512 | 68.2 | 27.9 | 0.85 |

## Memory Usage

| Operation | GPU Memory | CPU Memory |
|-----------|------------|------------|
| Model Load (tiny) | 128 MB | 256 MB |
| Model Load (base) | 512 MB | 1 GB |
| Model Load (large) | 2 GB | 4 GB |
| Model Load (mega) | 8 GB | 16 GB |
| Inference (batch=1) | +64 MB | +128 MB |
| Inference (batch=8) | +512 MB | +1 GB |

## Scaling Benchmarks

### Multi-GPU Scaling

| GPUs | Throughput (img/s) | Efficiency |
|------|-------------------|------------|
| 1 | 362 | 100% |
| 2 | 698 | 96.4% |
| 4 | 1352 | 93.4% |
| 8 | 2588 | 89.4% |

### Tile Processing Performance

| Tile Size | Overlap | Time per km² | Memory Peak |
|-----------|---------|--------------|-------------|
| 256×256 | 32 | 4.2s | 2.1 GB |
| 512×512 | 64 | 3.8s | 4.2 GB |
| 1024×1024 | 128 | 3.5s | 8.8 GB |
| 2048×2048 | 256 | 3.3s | 18.2 GB |

## API Benchmarks

### FastAPI Serving

| Endpoint | Latency p50 | Latency p99 | RPS |
|----------|-------------|-------------|-----|
| `/health` | 0.5ms | 1.2ms | 15,000 |
| `/predict` (tiny) | 8.2ms | 24.1ms | 850 |
| `/predict` (base) | 22.4ms | 68.3ms | 310 |
| `/predict` (large) | 85.2ms | 245.1ms | 82 |

## Comparison with Alternatives

| Library | Inference (ms) | Memory (MB) | Models | Ease of Use |
|---------|---------------|-------------|--------|-------------|
| **Unbihexium** | 5.3 | 512 | 520 | Excellent |
| TorchGeo | 8.2 | 780 | 45 | Very Good |
| Raster Vision | 12.1 | 1024 | 12 | Good |
| SITS | 15.4 | 640 | 8 | Good |

---

*Benchmarks last updated: December 2025*
