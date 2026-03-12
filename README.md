# ECM_2526_FinalProject_Gr12
### Predicting Investment Potential and Ethical Risk in Pharmaceutical Startups

<p align="center">

<a href="https://github.com">
<img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github">
</a>

<a href="https://www.kaggle.com">
<img src="https://img.shields.io/badge/Kaggle-Datasets-20BEFF?style=for-the-badge&logo=kaggle">
</a>

<a href="https://scikit-learn.org">
<img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikitlearn">
</a>

<a href="https://pytorch.org">
<img src="https://img.shields.io/badge/NLP-Transformers-red?style=for-the-badge&logo=pytorch">
</a>

</p>

---

# Project Overview

This project explores the use of **Machine Learning to evaluate pharmaceutical startups for investment purposes**, combining two key dimensions:

- **Commercial viability**
- **Ethical and ESG risk**

Biotechnology innovation represents one of the most promising yet uncertain sectors for investors. Pharmaceutical startups often operate in highly complex regulatory environments and face ethical scrutiny related to clinical trials, accessibility of medicines, and societal impact.

The objective of this project is to build a **data-driven decision support system** capable of assessing both **investment potential and ethical risk**.

The system is designed primarily for **investment funds focusing on ethical biotech innovation**.

---

# Teaching & Supervision

This project was conducted within the **Data Science curriculum at École Centrale Méditerranéenne**, in the **DDEFI specialization track**.

### Course Instructor

[Sitraka Matthieu FORLER](https://www.linkedin.com/in/sitraka-matthieu-forler/)  
Senior Data Scientist  
Professor of Applied Machine Learning

---

# Project Team

## Team Members

### Jajar26
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/Jajar26)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?logo=kaggle)](https://www.kaggle.com/jajar26)

### audreynrr
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/audreynrr)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?logo=kaggle)](https://www.kaggle.com/audreynrr)

### NinoUAM
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/NinoUAM)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?logo=kaggle)](https://www.kaggle.com/ninouam)

### mailisbrs
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/mailisbrs)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?logo=kaggle)](https://www.kaggle.com/mailisbrs)

---

# Problem Statement

Evaluating pharmaceutical startups presents several challenges for investors:

- Information is **fragmented across many sources**
- **Ethical concerns are difficult to quantify**
- Biotech innovation involves **high uncertainty and long R&D cycles**
- Many startups **fail before reaching commercialization**

Traditional financial analysis alone is therefore insufficient.

The goal of this project is to develop a **machine learning system capable of predicting both commercial viability and ethical risk** associated with pharmaceutical startups.

The project seeks to answer the following key question:
**How can we anticipate the financial success of a pharmaceutical startup while ensuring that it acts ethically, in order to optimize capital allocation decisions?**

---

# Value Proposition

The proposed system provides investors with:

- Centralized analysis of **startup data**
- Quantified **ethical risk scoring**
- **ML-based predictions** of startup value
- **Decision support tools** for ethical investment strategies

This framework integrates both **financial indicators and ESG considerations** into a unified evaluation system.

---

# Why Machine Learning

Machine learning is necessary because:

- Data sources are **large and heterogeneous**
- Information evolves **dynamically over time**
- Ethical signals are embedded in **textual information**
- Relationships between features are **complex and nonlinear**

ML models are therefore well suited to uncover **hidden patterns across financial, scientific, and textual datasets**.

---

# Dataset Construction

The team constructs a **structured dataset combining multiple information sources**, including:

**Startup Metadata**

- founding year
- headquarters location
- founders
- research area

**Scientific Publications**

- related research papers
- citation metrics
- scientific impact

**ESG Indicators**

- governance metrics
- environmental impact indicators
- social responsibility indicators

**Financial Data**

- funding rounds
- venture capital investment
- valuation history

**Textual Data**

- news articles
- public reports
- research documents

These sources are **cleaned, standardized, and merged into a unified dataset**.

---

# Models

The system relies on **two core machine learning components**.

---

# Model 1 — Ethical Scoring

The first model uses **ESG-BERT**, a transformer-based NLP model designed to analyze **ESG-related textual content**.

Input sources include:

- news coverage
- public reports
- scientific publications
- regulatory documents

The model produces an **ethical score** for each startup.

This score quantifies potential risks related to:

- clinical ethics
- governance practices
- environmental responsibility
- access to healthcare.

---

# Model 2 — Investment Prediction

The second model predicts the **investment potential of each startup**.

Key outputs:

- predicted **future valuation**
- **probability of successful funding**

Input features include:

- financial indicators
- startup characteristics
- innovation metrics
- **ethical score generated by ESG-BERT**

This architecture integrates **financial and ethical dimensions into a single predictive framework**.

---

# Pipeline Architecture

The project follows a structured **Machine Learning pipeline**:

1. **Data Collection**  
   Aggregation of startup, financial, and ESG datasets

2. **Data Cleaning**  
   Handling missing values, inconsistencies, and duplicates

3. **ESG Text Analysis**  
   NLP processing of textual documents

4. **Ethical Scoring**  
   Ethical score generation using ESG-BERT

5. **Feature Engineering**  
   Creation of structured predictive features

6. **ML Prediction Model**  
   Training models to predict investment success

7. **Investment Recommendation Layer**  
   Decision-support output for investors

---

# Expected Outputs

The system produces:

- **Ethical Risk Score**
- **Predicted Future Startup Value**
- **Probability of Funding Success**
- **Investment Decision Indicators**

These outputs support **ethical venture capital decision-making in biotech**.

---

# Limitations

Several limitations should be considered:

- Startup data may be **incomplete**
- ESG scoring may introduce **model bias**
- Predictions remain **probabilistic**
- Biotech markets are inherently **volatile**

Therefore, the system should be used as a **decision-support tool rather than a fully automated investment engine**.

---

# Future Improvements

Potential extensions include:

- integration with **larger ESG datasets**
- linking with **biotech innovation databases**
- advanced **transformer models for scientific literature**
- **reinforcement learning approaches** for investment strategy optimization
- interactive **startup analytics dashboards**

---
