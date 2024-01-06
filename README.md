# mlops_g22_2024

This repository contains the project work carried out by group 22 for the MLOps course at DTU, January 2024. 

**Project authors:** <br>
Eszter Hetzmann (student id) <br>
Jakub Solis (student id) <br>
Malene Hornstrup Jespersen (s237246) <br>
Mathias Nissen (student id) <br>
Riccardo Conti (student id) <br>


1. **Overall goal of the project:**
The goal of this project is to perform multiclass classification to identify an expressed emotion based on a single sentence. 

2. **What framework are you going to use and you do you intend to include the framework into your project?**
Since the task is natural language processing (NLP), we use the [Transformers](https://github.com/huggingface/transformers) framework from the Huggingface group, which contains pretrained state-of-the-art models for NLP tasks. 

3. **What data are you going to run on (initially, may change)?**
The pretrained model is finetuned on the open source Kaggle dataset [Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp/data). The complete dataset contains 20,000 observations, divided into training (16,000), testing (2,000) and validation (2,000) subsets. Each observation contains a single text sentence and a corresponding label indicating one of six expressed emotions: sadness, anger, fear, joy, love, or surprise. 

4. **What models do you expect to use?**
We expect to use the pre-trained model DistilBERT from the Transformers framework finetuned on the emotions data.



## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── mlops_g22_2024  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
