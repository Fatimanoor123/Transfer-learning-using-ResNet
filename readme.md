# Transfer Learning using ResNet for Celebrity Recognition

## Description
This project demonstrates transfer learning using a pretrained ResNet18 model. The final layer is modified to recognize 10 categories (simulating Bengali celebrities) instead of the original 1000 ImageNet categories. The model is trained on the CIFAR-10 dataset as a practice dataset — the same concept applies directly to the Bengali Celebrity Multimodal Dataset project.

## What is Transfer Learning?
Transfer learning means borrowing a pretrained model that already knows edges, shapes, textures and face patterns from millions of images. We only replace the final layer to match our task. This saves months of training time and gives better results.

## Pipeline
```
Load Pretrained ResNet18 → Replace Final Layer (1000 → 10) → 
Train on CIFAR-10 → Evaluate on Test Data
```

## Model Architecture
```
ResNet18 (pretrained on ImageNet)
│
├── Conv layers 1-50  → already learned edges, shapes, patterns
│                        BORROWED from ImageNet training ✅
│
└── Final layer       → REPLACED
    Original : Linear(512, 1000)  → 1000 ImageNet categories
    Modified : Linear(512, 10)    → 10 celebrities
   
```

## Results
```
Training Accuracy : 69.8%  (after 2 epochs, 500 images)
Test Accuracy     : 52.0%  (on 200 unseen images)
Random Guessing   : 10.0%  (10 categories)

Transfer learning is 5x better than random guessing!
```

## Two Types of Transfer Learning
- **Feature Extraction** → freeze all layers, train only final layer → good for small datasets
- **Fine Tuning** → train all layers, pretrained weights = starting point → good for large datasets (used in Bengali celebrity project)

## Project Connection
This project directly simulates the Bengali Celebrity Multimodal Dataset pipeline:

- **CIFAR-10 categories** → simulates Bengali celebrities
- **ResNet18 pretrained** → borrows face pattern knowledge
- **Final layer modified** → Linear(512, 250) for 250 Bengali celebrities
- **Training loop** → same loop used for real celebrity recognition

## Tech Stack
- Python
- PyTorch
- TorchVision
- ResNet18 (pretrained)
- CIFAR-10 Dataset
- Google Colab

## How to Run
1. Open notebook in Google Colab
2. Run all cells in order
3. Model downloads automatically
4. CIFAR-10 dataset downloads automatically
5. Training starts and shows accuracy per epoch

## Requirements
```
pip install torch
pip install torchvision
```

## Key PyTorch Concepts Used
- `models.resnet18(weights=DEFAULT)` → load pretrained model
- `model.fc = nn.Linear(512, 10)` → replace final layer
- `nn.CrossEntropyLoss()` → loss for classification
- `torch.optim.Adam()` → optimizer for fine tuning
- `model.eval()` → switch to evaluation mode
- `torch.no_grad()` → disable gradient during testing

## Author
Fatima Noor
## Related Projects
- [face-detection-scrfd](https://github.com/fatima-noor-ai/face-detection-scrfd) → Face detection on images
- [video-face-detection-scrfd](https://github.com/fatima-noor-ai/video-face-detection-scrfd) → Face detection on videos
- [audio-processing-mel-spectrogram](https://github.com/fatima-noor-ai/audio-processing-mel-spectrogram) → Audio processing
