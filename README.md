# EfficientNetB0 Lightweight COVID-19 Detection from CT and X-ray Images

This repository provides a complete implementation for a lightweight and accurate deep learning model to classify COVID-19 from chest X-ray and CT images. The model is based on EfficientNetB0 and is optimized using pruning and quantization-aware training (QAT) to reduce storage size and parameter count without significantly compromising accuracy.

## Features

- Based on EfficientNetB0 pretrained on ImageNet
- 98% pruning to reduce model complexity
- Quantization-aware training for 8-bit deployment
- Final model size approximately 106 KB
- Supports both 2-class and 3-class classification

## Dataset Structure

Your dataset folder should be structured as follows:

dataset_path/
- COVID-19/
- Normal/
- Pneumonia/
## How to Run

1. Update the `DATASET_PATH` variable in `efficientnetb0_qat.py` with your dataset path.
2. Run the script:


## Output Files

- `efficientnetb0_covid_qat_model.h5`: Trained and quantized Keras model
- `efficientnetb0_covid_qat_model.tflite`: Optimized model for deployment on edge/mobile devices

## Performance Summary

| Model                  | Accuracy (3-Class) | Parameters (Millions) | Storage Size |
|------------------------|-------------------|------------------------|---------------|
| EfficientNetB0 Original | 85.1%              | 5.3                    | 424 KB        |
| Pruned + QAT Model     | 98.15%             | 0.11                   | 106 KB        |

## Citation

If you use this code or refer to this work, please cite the following paper:

Sriwiboon, N. (2025). Efficient and lightweight CNN model for COVID-19 diagnosis from CT and X-ray images using customized pruning and quantization techniques. *Neural Computing and Applications*. https://doi.org/10.1007/s00521-025-11219-0

## Author

Nattavut Sriwiboon  
Department of Computer Science and Information Technology  
Kalasin University, Thailand  
Email: nattavut.sr@ksu.ac.th

## License

This project is licensed under the MIT License.
