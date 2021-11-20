# Tensorflow Developer Certificate 2022: Zero to Mastery

Course notebooks and milestone projects for Daniel Bourke's [Udemy Course](https://pwcanalytics.udemy.com/course/tensorflow-developer-certificate-machine-learning-zero-to-mastery).

Milestone Projects are as follows:
1. FoodVision - Classification of Kaggle's Food101 Dataset.
2. SkimLit - Reconstructing & reformatting PudMed abstracts to make them more readable.
3. BitPredict - Time-series forecasting the price of bitcoin.


## 1. FoodVision

Dataset: [Food 101](https://www.kaggle.com/dansbecker/food-101)

Models:
- Baseline feature extractor based on EfficientNet B0, trained using mixed precision training: Accuracy 70%.
- Fine-tuned EfficientNet B0 last 3 layers: Accuracy 74%.
- Fine-tuned EfficientNet B0 last 3 layers and using an augmented dataset: Accuracy 79%.
- Fine-tuned EfficientNet B0 all layers while using an augmented dataset, and an adpative learning rate: Accuracy 79%.


## 2. SkimLit

Dataset: [PudMed RCT](https://github.com/Franck-Dernoncourt/pubmed-rct.git)

ETL:
- Preprocess each abstract and append into a dataframe with a line per sentennce in abstract & relevant metadata (line number, total number of lines & classification (e.g. objective, method, conclusion etc))).
- One hot encode label values, line numbers and total lines.
- Vectorise and create embeddings (USE, GloVe, BERT, custom embedding layers).
- Batch and prefect datasets to optimise training speeds.

Models:
- Basline TF-IDF Multinomial Naive-Bayes Classifier: Accuracy 72%
- Conv1D (Token embedding layer): Accuracy 79%.
- Feature Extractor (USE embeddings): Accuracy 71%.
- Conv1D (Character-level embeddings): Accuracy 61%.
- Conv1D (Hybrid embedding layer: Token & Character level): Accuracy 72%.
- Transfer learning (Tribrid embedding layers: Token, Character, and Positional Embeddings): Accuracy 84%.

## 3. BitPredict

Dataset: [BTC data](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv)

Models:
- Niave-Bayes Classifier.
- Dense (Window size: 7 days, Horizon: 1 day).
- Dense (Window size: 7 days, Horizon: 1 day).
- Dense (Window size: 30 days, Horizon: 1 day).
- Dense (Window size: 30 days, Horizon: 7 days).
- 1D CNN.
- Bi-directional LSTM.
- Multivariate Dense.
- NBEATS Algorithm.
- Ensemble Model (Series of Dense models using NBEATS data pipeline).

[Certificate of Course Completion](https://pwcanalytics.udemy.com/certificate/UC-7c596c9d-505b-44ff-abf8-f46ee10c8877/)
