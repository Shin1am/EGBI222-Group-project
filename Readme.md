# YouTube Video Category Classification (Bag-of-Words, ML)

This Project Classify YouTube videos into categories using classic Machine Learning and a Bag-of-Words (BoW) text representation built transcribed audio.
This repository demonstrates a full NLP pipeline: data prep → text cleaning → BoW/TF-IDF vectorization → model training (e.g., Multinomial Naive Bayes / Logistic Regression / Linear SVM) → evaluation.

⁺˚⋆｡°✩₊✩°｡⋆˚⁺⁺˚⋆｡°✩₊✩°｡⋆˚⁺⁺˚⋆｡°✩₊✩°｡⋆˚⁺⁺˚⋆｡°✩₊✩°｡⋆˚⁺⁺˚⋆｡°✩₊✩°｡⋆˚⁺

---

## Key Features

- Data Preprocessing: Download, transcribe and translated the audio based on youtube-id.

- Vectorization: CountVectorizer (BoW) and TF-IDF options.

- Models: Multinomial Naive Bayes, Logistic Regression, Linear SVM (easily extensible).

- Evaluation: accuracy, macro/micro F1, confusion matrix, classification report.

- CLI & Notebook: run end-to-end from command line or explore via notebooks.

---

## Project Structure ദ്ദി(ᵔᗜᵔ)

```bash
.
├── dataset/
│   └── youtube-videos-data-for-ml-and-trend-analysis/
│        ├── youtube_data.csv        # Original dataset
│        ├── final_df.csv            # Preprocessed dataset (translated + labeled)
│
├── EGBI222_project_2.ipynb          # Main notebook (end-to-end workflow)
│
└── README.md
```
---

## Problem & Data
- ### Task: Multiclass classification of YouTube video category (label) from text features (transcribed audio).

- ### Expected Columns in the input CSV (example):

    - video_id (optional)

    - transcribed_audio (string)

    - translated_audio (string)

    - category (string or int label)

You can map YouTube’s official category IDs to human-readable names or use your own label schema.

---

# Workflow Overview
1. **Download Audio**  
   Extract audio from YouTube using `yt-dlp` based on video IDs.

2. **Transcription**  
   Use **Whisper** to convert audio into text (speech-to-text).

3. **Translation**  
   Apply **Deep Translator** to ensure all text is in English.

4. **Preprocessing**  
   - Lowercasing, punctuation & stopword removal  
   - Lemmatization (optional)

5. **Vectorization & Modeling**  
   Build Bag-of-Words and TF-IDF representations for text, then train models.

6. **Evaluation**  
   Compare model accuracy and F1 scores, visualize confusion matrices.

---

## 🧰 Requirements

You can run the notebook directly — no additional scripts needed.  
Make sure these Python packages are installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn yt-dlp deep-translator openai-whisper
```


## ˚₊· ͟͟͞͞➳❥ How to run

You can execute this project either directly in **Google Colab** or by running it locally.

---

### 🧭 Option 1: Run in Google Colab

You can open and run the full notebook directly in Colab by clicking the badge below:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tr1mE7DNLKKROcIXu3vYRd2VNj13iTQB?usp=sharing)

This will launch an interactive environment with all dependencies preinstalled.  
Simply run the notebook cells in order to download audio, transcribe, translate, and train the models.

---

### 💻 Option 2: Run Locally

1. **Clone the repository**

```bash
   git clone https://github.com/Shin1am/EGBI222-Group-project.git
   cd EGBI222-Group-project
   ```

2. **Open the Notebook**

```bash
   jupyter notebook EGBI222_project_2.ipynb
   ```

3. **Set your dataset path**

   Update the dataset path in the notebook before running the preprocessing section:

```py
   df = pd.read_csv("your-path-to-csv")
```

4. **Run all cells sequentially**

    The notebook will:

        - Download audio using yt-dlp

        - Transcribe with Whisper

        - Translate with Deep Translator

        - Train and evaluate multiple ML models

## Contributors

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