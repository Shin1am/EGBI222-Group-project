# YouTube Video Category Classification (Bag-of-Words, ML)

This Project Classify YouTube videos into categories using classic Machine Learning and a Bag-of-Words (BoW) text representation built transcribed audio.
This repository demonstrates a full NLP pipeline: data prep → text cleaning → BoW/TF-IDF vectorization → model training (e.g., Multinomial Naive Bayes / Logistic Regression / Linear SVM) → evaluation.

⁺˚⋆｡°✩₊✩°｡⋆˚⁺⁺˚⋆｡°✩₊✩°｡⋆˚⁺⁺˚⋆｡°✩₊✩°｡⋆˚⁺⁺˚⋆｡°✩₊✩°｡⋆˚⁺⁺˚⋆｡°✩₊✩°｡⋆˚⁺

## Key Features

- Data Preprocessing: Download, transcribe and translated the audio based on youtube-id.

- Vectorization: CountVectorizer (BoW) and TF-IDF options.

- Models: Multinomial Naive Bayes, Logistic Regression, Linear SVM (easily extensible).

- Evaluation: accuracy, macro/micro F1, confusion matrix, classification report.

- CLI & Notebook: run end-to-end from command line or explore via notebooks.

## Project Structure ദ്ദി(ᵔᗜᵔ)

.
├─ dataset/
│  ├─ youtube-videos-data-for-ml-and-trend-analysis/               
│       ├─ youtube_data.csv    # original CSV
|       ├─ final_df            # final dataset that have been modified.
│    
├─ EGBI222_project_2.ipynb      # jupyter notebook for this project
│   
└─ README.md


## Problem & Data
- ###Task: Multiclass classification of YouTube video category (label) from text features (transcribed audio).

- ###Expected Columns in the input CSV (example):

    - video_id (optional)

    - transcribed_audio (string)

    - translated_audio (string)

    - category (string or int label)

You can map YouTube’s official category IDs to human-readable names or use your own label schema.

## Setup (⸝⸝> ᴗ•⸝⸝)
### 1) Clone & create environment
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Using venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## requirements
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
tqdm
nltk

import nltk; nltk.download("stopwords"); nltk.download("wordnet"); nltk.download("omw-1.4")



## ˚₊· ͟͟͞͞➳❥ start 

###Put your data
data/raw/youtube_metadata.csv

###1) (Optional) Split data

If your repo includes a splitter script, run it; otherwise most train.py scripts accept a single CSV and do an internal train/val/test split.

###2) Train (BoW baseline)
python -m src.train \
  --input_csv data/raw/youtube_metadata.csv \
  --text_cols title description \
  --label_col category \
  --vectorizer bow \
  --ngram 1 2 \
  --min_df 2 \
  --model multinb \
  --save_dir models/


###3) Evaluate
If train.py already prints metrics, you set. Otherwise:
python -m src.evaluate \
  --model_path models/model.joblib \
  --vectorizer_path models/vectorizer.joblib \
  --eval_csv data/processed/test.csv \
  --text_cols title description \
  --label_col category

###4) Predict (single text or batch)
#### Single example
python -m src.predict \
  --model_path models/model.joblib \
  --vectorizer_path models/vectorizer.joblib \
  --title \
  --description

#### Batch (CSV with title/description)
python -m src.predict \
  --model_path models/model.joblib \
  --vectorizer_path models/vectorizer.joblib \
  --input_csv data/raw/unlabeled.csv \
  --output_csv data/predictions.csv \
  --text_cols title description


##Preprocessing
Typical cleaning applied in src/preprocess.py:

- Lowercase

- Strip URLs, HTML, emojis, and punctuation

- Remove digits (optional)

- Remove stopwords

- Lemmatize (optional)

- Join title + description (and optionally tags)

Configure these via flags or a small config in src/config.py.

##Models

Multinomial Naive Bayes: strong BoW baseline, fast and robust on sparse text.

Logistic Regression (One-vs-Rest): good accuracy, interpretable weights.

Linear SVM (SVC with linear kernel): strong linear baseline on TF-IDF.

Try both BoW and TF-IDF; often TF-IDF + Linear SVM gives a strong classic baseline.

##Metrics

- During training/evaluation you’ll see:

- Accuracy

- Macro / Micro F1-score

- Confusion matrix

- Per-class precision/recall (classification report)

##Reproducibility

- Fixed seeds (e.g., --seed 42) for splits and model initialization.

- Save both model.joblib and vectorizer.joblib for consistent inference.

- Log hyperparameters in a run config or artifact filename.

##Example Commands
### Multinomial NB + BoW
python -m src.train --vectorizer bow --model multinb --ngram 1 2 --min_df 2

### Logistic Regression + TF-IDF
python -m src.train --vectorizer tfidf --model logreg --ngram 1 2 --C 2.0 --max_iter 200

### Linear SVM + TF-IDF (strong baseline)
python -m src.train --vectorizer tfidf --model svm --ngram 1 2 --C 1.0

#Notebook
Open notebooks/01_eda_and_baseline.ipynb for:

EDA (label distribution, text lengths)

Quick BoW/TF-IDF baselines

Confusion matrix plots

##Checklist ⸜(｡˃ ᵕ ˂ )⸝♡

 Data placed in data/raw/…

 Preprocessing flags set (stopwords, ngrams, min_df)

 Vectorizer & model trained and saved in /models

 Evaluation run on held-out test set

 predict.py tested on sample lines

 
##Contributors

Yada Yimngam 6713359
Nicharee Nunuan 6713363
Pannawat Thongpron 6713376
Pheemchayut Matures 6713387
Lapasrada Tiatrakul 6713389

                      ♡   ╱|、
                          (˚ˎ 。7  
                           |、˜〵          
                          じしˍ,)ノ


 ꒷꒦꒷꒦꒷꒦꒷꒦꒷꒦꒷꒷꒦꒷꒦꒷꒦꒷꒦꒷꒦꒷꒷꒦꒷꒦꒷꒦꒷꒦꒷꒦꒷꒷꒦꒷꒦꒷꒦꒷꒦꒷꒦꒷꒷꒦꒷꒦꒷꒦꒷꒦꒷꒦꒷꒷꒦꒷꒦꒷꒦꒷꒦꒷꒦꒷꒷꒦꒷꒦꒷꒦꒷꒦꒷꒦꒷