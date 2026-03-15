ECM_2526_FinalProject_Gr12
Predicting Financial Success and Ethical Risk in Pharmaceutical Startups
<p align="center"> <a href="https://github.com"> <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github"> </a> <a href="https://www.kaggle.com"> <img src="https://img.shields.io/badge/Kaggle-Datasets-20BEFF?style=for-the-badge&logo=kaggle"> </a> <a href="https://scikit-learn.org"> <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikitlearn"> </a> <a href="https://pytorch.org"> <img src="https://img.shields.io/badge/NLP-Transformers-red?style=for-the-badge&logo=pytorch"> </a> </p>
Project Overview

This project develops a machine learning system designed to support investment decisions in pharmaceutical startups.

Investing in biotech companies is particularly challenging due to:

long R&D cycles

high failure rates in drug development

complex regulatory environments

limited transparency on early-stage companies

To address these challenges, our system combines financial prediction models with automated ethical analysis based on ESG criteria.

The goal is to provide a data-driven framework to evaluate both the financial potential and the ethical risk of biotech startups.

Academic Context

This project was conducted as part of the Data Science curriculum at École Centrale Méditerranéenne, within the DDEFI specialization track.

Instructor

Sitraka Matthieu FORLER

Senior Data Scientist — Professor of Applied Machine Learning

Project Team
Hajar Belgroun

GitHub
 • Kaggle

Audrey Nourry

GitHub
 • Kaggle

Nino Tissot

GitHub
 • Kaggle

Mailis Briens

GitHub
 • Kaggle

Problem Statement

Evaluating the future success of pharmaceutical startups is difficult because most early-stage companies lack long-term performance indicators.

In addition, ethical considerations are increasingly important in healthcare innovation, particularly regarding:

clinical trial transparency

governance practices

access to treatments

societal impact of medical technologies

This project explores the following question:

Can machine learning help anticipate the financial success of pharmaceutical startups while integrating ethical risk into the evaluation process?

Dataset

The project relies on the Crunchbase venture capital dataset, which includes 54,294 startups across all sectors.

After filtering for pharma and biotech related companies, the dataset contains:

4,207 startups in the sector

225 acquisitions

147 failures

3,730 operating companies

A data audit also revealed 108 startups labeled as "operating" despite having IPO-related financial indicators.
These cases were reclassified as successful exits, increasing the number of training examples.

To prevent data leakage, the model is trained only on startups with known outcomes before January 1st, 2012, and evaluated on companies still operating afterward.

Feature Engineering

Instead of using raw variables, several investor-oriented financial indicators were created, including:

startup age

time to first funding

total fundraising duration

funding speed

funding regularity

Additional binary features indicate the presence of different funding types:

seed

venture

grant

round A

round B

Missing values are handled using IterativeImputer for most models, while HistGradientBoosting handles them natively.

Machine Learning Models

Four models were trained and evaluated:

Logistic Regression

Random Forest

HistGradientBoosting

Support Vector Machine (SVM)

To address class imbalance, all models use:

class_weight = "balanced"

The final predictor uses a StackingClassifier, combining the outputs of the base models using a logistic regression meta-model.

Performance

Final model performance:

F1-score: 80.5%

Recall: 96%

Cross-validation F1: 78.6% ± 3%

These results indicate strong ability to identify successful startups while maintaining generalization.

Ethical Analysis

Beyond financial prediction, the project introduces an automated ethical scoring component.

For selected startups, textual data is collected from:

ClinicalTrials.gov (clinical trials)

PubMed (scientific publications)

company websites (mission and governance)

Texts are analyzed using ESG-BERT, a transformer-based model specialized in ESG classification.

A sentiment analysis step using DistilBERT adjusts scores based on the tone of ESG-related mentions.

To account for missing information, a data quality score weights the reliability of each ethical evaluation.

Final Scoring Framework

The final investment score combines financial and ethical components:

Final Score = 70% Financial Score + 30% Ethical Score

This weighting reflects a balanced investment strategy prioritizing financial potential while accounting for ethical risks.

Among the analyzed companies, DistalMotion emerges as the top candidate based on this combined evaluation.

Deployment

To make the system usable in practice, the models are deployed through a FastAPI service.

Available endpoints include:

financial scoring

combined financial + ethical scoring

batch scoring (up to 100 startups)

An interactive dashboard allows analysts to explore startup evaluations directly from a web interface.

Limitations

Several limitations should be considered:

startup data may be incomplete or outdated

ethical scoring depends on available textual sources

long-term startup success remains inherently uncertain

The system is therefore intended as a decision-support tool, not as a fully automated investment engine.

Future Improvements

Possible extensions include:

expanding the dataset with post-2012 startup outcomes

scaling ESG analysis to all startups instead of the top 20

empirically optimizing the financial/ethical weighting

improving NLP coverage of scientific literature
