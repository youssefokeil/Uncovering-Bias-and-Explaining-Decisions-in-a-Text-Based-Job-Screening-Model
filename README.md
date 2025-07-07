# BERT Hiring Model: Bias Detection and Mitigation

This project investigates gender bias in a BERT-based hiring classification model and implements counterfactual data augmentation to reduce bias while maintaining model performance.

## Overview

The project demonstrates how representation imbalance in training data can lead to gender bias in hiring decisions, and shows how counterfactual data augmentation can effectively mitigate these biases.

## Methodology

### 1. Dataset Preparation
- **Intentional Bias Introduction**: Created a training/test split with severe representation imbalance
  - 95% of male candidate data included in training set
  - Only 10% of female candidate data included in training set
- **Tokenization**: Used BERT tokenizer to convert text inputs into model-ready tokens

### 2. Model Architecture
- **Base Model**: BERT-based classifier using `AutoModelForSequenceClassification` from Hugging Face Transformers
- **Task**: Binary classification for hiring decisions (Hire/No-Hire)

### 3. Explainability Analysis
- **Tool**: SHAP (SHapley Additive exPlanations) for model interpretation
- **Scope**: Analyzed 5 model predictions (3 Hire, 2 No-Hire decisions)
- **Findings**: 
  - Gender-indicative words showed minor but measurable influence on predictions
  - Gender correlation bias was lower than initially expected
![SHAp](https://github.com/youssefokeil/Uncovering-Bias-and-Explaining-Decisions-in-a-Text-Based-Job-Screening-Model/blob/main/images/beforeCDA-SHAP.png)

### 4. Bias Evaluation Metrics
The model was evaluated using standard fairness metrics:
- **Demographic Parity Difference**
- **Equal Opportunity** (True Positive Rate by gender)
- **True Positive Rate Difference**
- **False Positive Rate Difference**
- **Average Odds Difference**

## Key Findings

### Before Mitigation (Representation Imbalance)
- **Demographic Parity Difference**: 10.6%
- **Equal Opportunity**: Male 70%, Female 76.7%
- **True Positive Rate Difference**: 6.73%
- **False Positive Rate Difference**: 8.01%
- **Average Odds Difference**: 7.37%

![BeforeCDA](https://github.com/youssefokeil/Uncovering-Bias-and-Explaining-Decisions-in-a-Text-Based-Job-Screening-Model/blob/main/images/beforeCDA.png)

### After Counterfactual Data Augmentation
- **Demographic Parity Difference**: 4.6% (↓ 56.6% improvement)
- **Equal Opportunity**: Male 80%, Female 78.7%
- **True Positive Rate Difference**: 1.29% (↓ 80.8% improvement)
- **False Positive Rate Difference**: 2.40% (↓ 70.0% improvement)
- **Average Odds Difference**: 1.84% (↓ 75.0% improvement)

![AfterCDA](https://github.com/youssefokeil/Uncovering-Bias-and-Explaining-Decisions-in-a-Text-Based-Job-Screening-Model/blob/main/images/afterCDA.png)

## Bias Mitigation Strategy

### Counterfactual Data Augmentation (CDA)
- **Method**: Systematic gender term swapping in training data
- **Examples**:
  - "he" ↔ "she"
  - "man" ↔ "woman"
  - "his" ↔ "her"
  - And other gender-specific terms

### Results
The CDA approach successfully reduced bias across all fairness metrics while maintaining model performance, demonstrating significant improvements in demographic parity and equal opportunity.

## Technical Stack

- **Framework**: Hugging Face Transformers
- **Model**: BERT (AutoModelForSequenceClassification)
- **Explainability**: SHAP
- **Bias Mitigation**: Counterfactual Data Augmentation

## Usage

1. **Data Preparation**: Ensure your dataset includes gender-indicative terms for comprehensive bias analysis
2. **Model Training**: Train BERT classifier with intentionally imbalanced data to observe bias
3. **Bias Evaluation**: Use the provided fairness metrics to quantify bias
4. **Mitigation**: Apply counterfactual data augmentation by swapping gender terms
5. **Re-evaluation**: Compare bias metrics before and after mitigation

## Key Takeaways

- **Representation matters**: imbalances in training data can lead to measurable bias
- **Explainability is crucial**: SHAP analysis helps identify specific tokens contributing to biased decisions
- **Counterfactual augmentation works**: Simple gender term swapping can significantly reduce bias
- **Multiple metrics needed**: Comprehensive bias evaluation requires multiple fairness metrics
