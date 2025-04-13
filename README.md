# Beyond Degradation Conditions: All-in-One Image Restoration via HOG Transformers

This repository contains the official implementation of the paper "Beyond Degradation Conditions: All-in-One Image Restoration via HOG Transformers". 

### Abstract

All-in-one image restoration, which aims to address diverse degradations within a unified framework, is critical for practical applications. However, existing methods rely on predicting and integrating degradation conditions, which can misactivate degradation-specific features in complex scenarios, limiting their restoration performance. To address this issue, we propose a novel all-in-one image restoration framework guided by Histograms of Oriented Gradients (HOG), named HOGformer. By leveraging the degradation-discriminative capability of HOG descriptors, HOGformer employs a dynamic self-attention mechanism that adaptively attends to long-range spatial dependencies based on degradation-aware HOG cues. To enhance the degradation sensitivity of attention inputs, we design a HOG-guided local dynamic-range convolution module that captures long-range degradation similarities while maintaining awareness of global structural information. Furthermore, we propose a dynamic interaction feed-forward module, efficiently increasing the model capacity to adapt to different degradations through channel–spatial interactions. Extensive experiments across diverse benchmarks, including adverse weather and natural degradations, demonstrate that HOGformer achieves state-of-the-art performance and generalizes effectively to complex real-world degradations.

### Framework
![motivation](https://github.com/user-attachments/assets/eee2809c-8c4c-40b3-afbc-2c03317c71bc)
![method2](https://github.com/user-attachments/assets/257fc0a2-fee3-4960-8d85-b5f45bf7ebda)

## Setup

### Installation

```bash
# Clone the repository
git clone https://github.com/Fire-friend/HOGformer.git
cd HOGformer

# Create and activate virtual environment (recommended)
conda create -n HOGformer python==3.10

# Install dependencies
pip install -r requirements.txt
```

## Dataset

### Data Preparation

#### Setting I

|      |      |
| ---- | ---- |
|      |      |

#### Setting II

|      |      |
| ---- | ---- |
|      |      |


## Usage

### Training

```bash
python XXXX
```

Key parameters:

- [Other important parameters]

### Evaluation

```bash
python XXXX
```

### Using Pretrained Models

We provide pretrained models that can be used directly for inference:

```bash
python XXXX
```

## Results
![图片8](https://github.com/user-attachments/assets/7bbd3a2d-6a88-4a7a-b1b8-ab7d9197541a)
![图片9](https://github.com/user-attachments/assets/3268651b-0581-4c92-b4db-0b8fe6038745)

Here are our experimental results on [dataset name]:

| Model      | Metric 1 | Metric 2 | Metric 3 |
| ---------- | -------- | -------- | -------- |
| Our Method | XX.X%    | XX.X%    | XX.X%    |
| Baseline 1 | XX.X%    | XX.X%    | XX.X%    |
| Baseline 2 | XX.X%    | XX.X%    | XX.X%    |

## Visualizations

XXXX

## Citation

If you use our code or method in your research, please cite our paper:

```
@article{lastname2023paper,
  title={Beyond Degradation Conditions: All-in-One Image Restoration via HOG Transformers},
  author={Jiawei Wu, Zhifei Yang, Zhe Wang, Zhi Jin},
  journal={},
  year={}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

## Contact

For any questions, please contact: [wujw97@mail2.sysy.edu.cn]

**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox, [Histoformer](https://github.com/sunshangquan/Histoformer), and [DiffUIR](https://github.com/iSEE-Laboratory/DiffUIR).
