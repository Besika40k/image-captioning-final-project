# Image Captioning — Final Project

A neural network system that generates natural language captions for images. The model takes an image as input and produces a descriptive sentence as output, bridging visual understanding and natural language generation.

## Team

- [Name 1] — GitHub: @Besika40k
- [Name 2] — GitHub: @Nok1on1

## Overview

This project implements an image captioning system trained on a dataset of 8,000 images, each paired with five human-written captions. The system uses an encoder-decoder architecture to map visual features to natural language descriptions.

## Project Structure

image-captioning-final-project/

│

├── README.md

├── .gitignore

├── requirements.txt

│

├── data/

│ └── (dataset — not tracked in git, see Setup)

│

├── notebooks/

│ ├── data_and_training.ipynb

│ └── inference.ipynb

│

├── src/

│ ├── dataset.py

│ ├── vocabulary.py

│ ├── model.py

│ ├── train.py

│ └── utils.py

│

└── checkpoints/

└── (trained model weights — not tracked in git, see Setup)

- `notebooks/data_and_training.ipynb` — data loading, model definition, training process, model saving
- `notebooks/inference.ipynb` — caption generation on unseen test images, with success and failure case analysis
- `src/` — modular source code (dataset handling, vocabulary, model architecture, training utilities)
- `checkpoints/` — trained model weights

## Data

This project uses the Flickr8k dataset, consisting of 8,091 unique images, each paired with 5 human-written captions (40,455 total image-caption pairs).

- **Format:** `data/caption_data/captions.txt` is a CSV file with a header row and two columns — `image` (filename) and `caption` (text). Each image filename appears on 5 consecutive rows, one per caption. Corresponding image files live in `data/caption_data/Images/`.
- **Vocabulary:** built from training captions only, using a simple regex tokenizer (`\w+`, lowercased) with a minimum frequency threshold of 5 occurrences. This keeps the vocabulary at a manageable size (2,982 words) while filtering out rare typos, names, and one-off compound words. Special tokens: `<pad>`, `<start>`, `<end>`, `<unk>`.
- **Train/val/test split:** 80/10/10, split at the image level (not the caption-row level) with a fixed random seed (42) for reproducibility. This ensures all 5 captions belonging to a given image stay within the same split, preventing data leakage between training and evaluation. Resulting split sizes:

| Split      | Images | Caption rows (approx.) |
|------------|--------|------------------------|
| Train      | 6,473  | ~32,365                |
| Validation | 809    | ~4,045                 |
| Test       | 809    | ~4,045                 |

The test split is held out entirely from training and model selection — it is used exclusively in `notebooks/inference.ipynb` for the final demonstration and success/failure analysis.

## Setup

1. Clone the repository:

```bash
   git clone https://github.com/your-username/image-captioning-final-project.git
   cd image-captioning-final-project
```

2. Create a virtual environment and install dependencies:

```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
```

3. Download the dataset:
   This project uses the Flickr8k dataset (`caption_data.zip`, provided by the course).
   Extract it so the structure matches:
   ```text
   data/
   └── caption_data/
       ├── Images/
       │   ├── 1000268201_693b08cb0e.jpg
       │   └── ...
       └── captions.txt
   ```
   `captions.txt` should have the header `image,caption`. If your extracted archive uses a different filename (e.g. `Flickr8k.token.txt`) or folder name (e.g. `Flicker8k_Dataset/`), rename/move it to match the structure above, or update the path passed to `load_captions_df()` in the notebooks.

4. Download pretrained model weights (optional, if not retraining from scratch):
   - placeholder for now, we will add the Google Drive / release link here once the model is trained

## How to Run

1. Open `notebooks/data_and_training.ipynb` to walk through data loading, model architecture, and training.
2. Open `notebooks/inference.ipynb` to generate captions on test images using the trained model.

placeholder for now, we will add more detailed run instructions (expected runtime, GPU requirements, etc.) here once the pipeline is finalized

## Model Architecture

placeholder for now, we will add a description of the encoder (e.g. CNN-based image feature extractor) and decoder (e.g. RNN/LSTM/Transformer-based caption generator) here once the architecture is implemented

## Training Details

placeholder for now, we will add details on training setup here — number of epochs, batch size, optimizer, loss function, hardware used, training time, etc.

## Results

placeholder for now, we will add quantitative results (e.g. loss curves, evaluation metrics) and qualitative examples (sample generated captions) here once training is complete

## Failure Case Analysis

placeholder for now, we will add examples of incorrect or nonsensical captions here, along with brief analysis of why the model failed on these cases

## Contributions

placeholder for now, we will add a breakdown of who worked on what (data pipeline, model architecture, training, inference, documentation) here

## License

placeholder for now, we will add a license here if needed (e.g. MIT) — optional for a course project
