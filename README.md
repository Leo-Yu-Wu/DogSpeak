# DogSpeak: Multi-Task Dog Audio Classification
DogSpeak is a deep learning framework designed to analyze dog vocalizations.
Using state-of-the-art audio transformers (Wav2Vec2-BERT), this system can identify individual dogs, classify breeds, and determine gender from raw audio clips.

## Key Features
1. *Multi-Task Learning*: Supports three distinct classification tasks:
    - *Dog Identification:* "Which specific dog is barking?"
    - *Dog Breed Classification:* "What breed is this?"
    - *Gender Classification:* "Is this a Male or Female?"
2. *Robust Data Pipeline:*
    - *Data Preprocessing:*
      1. drop dogs with less than 30 samples
      2. downsample to 16kHz
      3. trim silence so that audio clips are no longer than 4.5 seconds (covers 95% of audio clips)
    - *Corruption Handeling:*
      1. *Leak Prevention:* Uses `GroupShuffleSplit` to ensure the same dog never appears in both Training and Test sets for Breed/Gender tasks.
      2. *Corruption Handling:* Automatically detects and skips corrupted audio files during training without crashing.

## Installation
1. *Prerequisites:*
   - Linux or Windows (tested on Ubuntu 22.04 and Windows 11)
   - Python 3.10+
   - NVIDIA GPU (Recommended) with CUDA 11.8+
2. *Setup Environment:*
```bash
# Create a fresh environment
conda create -n dogspeak python=3.10
conda activate dogspeak

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas scikit-learn seaborn matplotlib tqdm pyyaml networkx==3.3 requests
```
alternatively, you can use `requirements.txt` to install all dependencies.
3. *Download Dataset:*
This project uses the [DogSpeak Dataset](https://huggingface.co/datasets/ArlingtonCL2/DogSpeak_Dataset) hosted on Hugging Face.
## Usage
1. *Configuration*  
The behavior of the model is controlled by YAML config files.  
    - `config.yaml` -> Defalut (Dog ID Task)
    - `config_breed.yaml` -> Bread Classification
    - `config_gender.yaml` -> Gender Classification
2. *Training*  
You can train any task using the generic train_task.py script.
*Train Breed Model:*  
```bash
python train_task.py config_breed.yaml
```
*Train Gender Model:*  
```bash
python train_task.py config_gender.yaml
```
*Train Dog ID Model:*  
```bash
python train_task.py config.yaml
```
## Evaluation
Generate detailed reports, calculate Macro F1-Score, and plot Confusion Matrices.  
*Basic Evaluation*
```bash
python evaluate.py config_breed.yaml
```
*Top-K Evaluation (For ID Task):* Since the ID task has 142+ classes, the confusion matrix can be messy. Use `--top_k` to plot only the most frequent dogs.
```bash
# Plot confusion matrix for the top 20 most active dogs
python src/evaluate.py config.yaml --top_k 20l
```
*Output:*
    - `checkpoints/confusion_matrix.png` -> Visual heatmap of model performance.
    - `checkpoints/evaluation_report.txt` -> Detailed precision/recall for every class.