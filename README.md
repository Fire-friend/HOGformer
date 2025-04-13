# Beyond Degradation Conditions: All-in-One Image Restoration via HOG Transformers

This repository contains the official implementation of the paper "Beyond Degradation Conditions: All-in-One Image Restoration via HOG Transformers". 

### Abstract

All-in-one image restoration, which aims to address diverse degradations within a unified framework, is critical for practical applications. However, existing methods rely on predicting and integrating degradation conditions, which can misactivate degradation-specific features in complex scenarios, limiting their restoration performance. To address this issue, we propose a novel all-in-one image restoration framework guided by Histograms of Oriented Gradients (HOG), named HOGformer. By leveraging the degradation-discriminative capability of HOG descriptors, HOGformer employs a dynamic self-attention mechanism that adaptively attends to long-range spatial dependencies based on degradation-aware HOG cues. To enhance the degradation sensitivity of attention inputs, we design a HOG-guided local dynamic-range convolution module that captures long-range degradation similarities while maintaining awareness of global structural information. Furthermore, we propose a dynamic interaction feed-forward module, efficiently increasing the model capacity to adapt to different degradations through channel–spatial interactions. Extensive experiments across diverse benchmarks, including adverse weather and natural degradations, demonstrate that HOGformer achieves state-of-the-art performance and generalizes effectively to complex real-world degradations.

### Framework



## Setup

### Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/[repository-name].git
cd [repository-name]

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Hardware Requirements

- Recommended: [List recommended hardware configurations, such as GPU type, memory requirements, etc.]
- Minimum: [List minimum hardware requirements]

## Dataset

### Data Preparation

[Explain how to obtain and prepare the dataset, including dataset download links, preprocessing steps, etc.]

```bash
# Example for data download
python scripts/download_data.py

# Example for data preprocessing
python scripts/preprocess_data.py --input_dir data/raw --output_dir data/processed
```

## Usage

### Training

```bash
python train.py --config configs/default.yaml
```

Key parameters:

- `--config`: Configuration file path
- `--batch_size`: Batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- [Other important parameters]

### Evaluation

```bash
python evaluate.py --model_path checkpoints/best_model.pth --test_data data/test
```

### Using Pretrained Models

We provide pretrained models that can be used directly for inference:

```bash
python inference.py --model_path pretrained/model.pth --input_file your_data.csv
```

## Results

Here are our experimental results on [dataset name]:

| Model      | Metric 1 | Metric 2 | Metric 3 |
| ---------- | -------- | -------- | -------- |
| Our Method | XX.X%    | XX.X%    | XX.X%    |
| Baseline 1 | XX.X%    | XX.X%    | XX.X%    |
| Baseline 2 | XX.X%    | XX.X%    | XX.X%    |

## Visualizations

[If you have visualization results, place some example images here with brief explanations]

## Citation

If you use our code or method in your research, please cite our paper:

```
@article{lastname2023paper,
  title={Paper Title},
  author={Lastname, Firstname and Coauthor, Name},
  journal={Journal/Conference Name},
  year={Publication Year}
}
```

## License

This project is licensed under the [License Type] License. See the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

## Contact

For any questions, please contact: [your.email@example.com]