# ECM_2526_FinalProject_Gr12  
### Predicting Financial Success and Ethical Risk in Pharmaceutical Startups

<p align="center"> <a href="https://github.com"> <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github"> </a> <a href="https://www.kaggle.com"> <img src="https://img.shields.io/badge/Kaggle-Datasets-20BEFF?style=for-the-badge&logo=kaggle"> </a> <a href="https://scikit-learn.org"> <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikitlearn"> </a> <a href="https://pytorch.org"> <img src="https://img.shields.io/badge/NLP-Transformers-red?style=for-the-badge&logo=pytorch"> </a> </p>
---

# Project Overview

This project explores how **machine learning can support investment decisions in pharmaceutical startups** by combining financial analysis with ethical evaluation.

Investing in biotech companies is particularly challenging. Development cycles are long, regulatory requirements are strict, and most drug candidates never reach the market. As a result, investors must make decisions under significant uncertainty.

To address this challenge, we develop a **data-driven decision support framework** capable of evaluating both the **financial potential** and the **ethical risk** of pharmaceutical startups.

The objective is not to replace human judgment, but to provide **analytical tools that help investors identify promising and responsible companies**.

---

# Academic Context

This project was conducted as part of the **Data Science curriculum at École Centrale Méditerranéenne**, within the **DDEFI specialization track**.

### Instructor

Sitraka Matthieu FORLER  
Senior Data Scientist  
Professor of Applied Machine Learning

---

# Project Team

Hajar Belgroun  
https://github.com/Jajar26  

Audrey Nourry  
https://github.com/audreynrr  

Nino Tissot  
https://github.com/NinoUAM  

Mailis Briens  
https://github.com/mailisbrs  

---

# Problem Statement

Evaluating pharmaceutical startups presents several challenges:

- early-stage companies have **limited historical data**
- biotech innovation involves **long research and development cycles**
- financial success often takes **many years to materialize**
- ethical concerns in healthcare are **increasingly important**

Traditional financial analysis alone is therefore insufficient.

This project investigates the following question:

**Can machine learning help anticipate the financial success of pharmaceutical startups while also accounting for ethical considerations?**

---

# Dataset

The project relies on the **Crunchbase venture capital dataset**, which includes **54,294 startups across multiple sectors**.

After filtering for companies related to **pharmaceuticals and biotechnology**, the dataset contains:

- 4,207 startups in the sector  
- 225 acquisitions  
- 147 failures  
- 3,730 companies still operating  

During data exploration, we identified **108 startups labeled as operating despite showing IPO-related financial indicators**.  
These companies were reclassified as successful exits, which increased the number of training examples.

To prevent **data leakage**, the model is trained only on startups whose outcomes were known **before January 1st, 2012**.

---

# Feature Engineering

Instead of relying on raw dataset variables, several **financial indicators inspired by investor reasoning** were constructed.

Key features include:

- startup age  
- time between founding and first funding  
- total fundraising duration  
- fundraising speed  
- fundraising regularity  

Additional **binary variables** indicate the presence of specific funding types:

- seed
- venture
- grant
- round A
- round B

Missing values are handled using **IterativeImputer** for most models, while **HistGradientBoosting** can handle missing values directly.

---

# Machine Learning Models

Four models were trained and compared:

- Logistic Regression  
- Random Forest  
- HistGradientBoosting  
- Support Vector Machine (SVM)

Because the dataset contains more successful startups than failures, all models use:

class_weight = "balanced"

to reduce the effect of class imbalance.

The final predictor uses a **StackingClassifier**, which combines the predictions of the base models using a logistic regression meta-model.

### Model Performance

Final results on the test set:

- **F1-score:** 80.5%  
- **Recall:** 96%  

Cross-validation results:

- **F1-score:** 78.6% ± 3%

These results indicate that the model successfully identifies most successful startups while maintaining good generalization.

---

# Ethical Analysis

Beyond financial prediction, the project introduces an **automated ethical evaluation component**.

Textual data is collected from several public sources:

- ClinicalTrials.gov (clinical trials)
- PubMed (scientific publications)
- company websites (mission and governance information)

These texts are analyzed using **ESG-BERT**, a transformer-based model specialized in ESG classification.

To account for context, a **sentiment analysis step using DistilBERT** adjusts ESG scores depending on whether mentions are positive or negative.

A **data quality score** is also used to account for missing sources.

---

# Final Scoring Framework

The final evaluation combines both dimensions:

Final Score = 0.7 × Financial Score + 0.3 × Ethical Score

This weighting reflects a strategy where **financial potential remains the primary driver**, while **ethical considerations influence the final ranking**.

Among the evaluated companies, **DistalMotion** appears as the most balanced candidate in terms of financial and ethical performance.

---

# Deployment

To make the system usable in practice, the models were integrated into a **FastAPI application**.

The API exposes several endpoints:

- financial scoring
- combined financial and ethical scoring
- batch scoring for multiple startups

An **interactive dashboard** allows users to explore results and test startups directly from a browser.

---

# Limitations

Several limitations should be considered:

- startup datasets may contain incomplete information  
- ethical scoring depends on available textual sources  
- predicting startup success over long horizons remains inherently uncertain  

The system should therefore be used as a **decision-support tool rather than a fully automated investment engine**.

---

# Future Work

Possible improvements include:

- extending the dataset with **more recent startup outcomes**
- scaling ESG analysis to **all startups instead of only the top candidates**
- empirically validating the **financial/ethical weighting**
- improving NLP coverage of **scientific literature and clinical data**

---
