# UVQ 1.5 TFLite: FLOAT32 vs INT8 Quantization Results
---
## Table 1: Performance (Inference Speed)

| **Metric**                  | **FLOAT32** | **INT8**  | **Improvement**      |
|:----------------------------|------------:|----------:|:--------------------:|
| **Average Inference Time**  | 15.235 s    | 11.750 s  | **22.9% faster**     |
| **Time per Frame**          | 0.802 s     | 0.618 s   | **22.9% faster**     |
| **Speedup Factor**          | 1.00x       | **1.30x** | **30% faster**       |
| **Time Saved per Video**    | -           | 3.485 s   | **22.9%**            |
| **Standard Deviation**      | 0.016 s     | 0.033 s   | Consistent           |

**Summary:** INT8 models deliver **1.30x speedup**, processing videos **22.9% faster** than FLOAT32.

---

## Table 2: Accuracy (Quality Assessment)

| **Metric**                      | **FLOAT32** | **INT8** | **Difference**     | **Status**       |
|:--------------------------------|------------:|---------:|:------------------:|:----------------:|
| **Overall Quality Score**       | 3.0440      | 3.1185   | 0.0744 (2.44%)     | ✅ Excellent     |
| **Mean Per-Frame Difference**   | -           | -        | 0.0869 (2.85%)     | ✅ Excellent     |
| **Max Per-Frame Difference**    | -           | -        | 0.1553 (5.13%)     | ✅ Acceptable    |
| **Score Standard Deviation**    | 0.0234      | 0.0245   | 0.0011             | ✅ Minimal       |

**Summary:** INT8 models maintain **excellent accuracy** with only **2.44% difference** in overall quality scores.

---

## Table 3: Model Size (Storage & Memory)

| **Component**        | **FLOAT32**  | **INT8**    | **Reduction** | **Savings**  |
|:---------------------|-------------:|------------:|:-------------:|:------------:|
| **ContentNet**       | 14.55 MB     | 4.27 MB     | **70.7%**     | 10.28 MB     |
| **DistortionNet**    | 14.53 MB     | 4.25 MB     | **70.7%**     | 10.28 MB     |
| **AggregationNet**   | 0.30 MB      | 0.11 MB     | **62.5%**     | 0.19 MB      |
| **Total Size**       | **29.38 MB** | **8.63 MB** | **70.6%**     | **20.75 MB** |

**Summary:** INT8 models are **70.6% smaller**, reducing total size from **29.38 MB to 8.63 MB**.

---
