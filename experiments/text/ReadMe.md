# Textual Model Experiments

This section contains all the experiments performed on our **textual stance classification models** for argument mining. The goal is to iteratively improve model performance, measured primarily by **F1-score (Binary - Positive Class)**.

---

## Experiments Overview

### 1. **PEFT (Parameter-Efficient Fine-Tuning)**
- **Purpose:** Fine-tune large language models (LLMs) efficiently using LoRA, AdaLoRA, and PromptTuning.
- **Output:** Fine-tuned adapters, performance metrics (accuracy, precision, recall, F1) per method.
- **Notes:** Useful to reduce computational resources while adapting LLMs to stance classification.

### 2. **Data Augmentation**
- **Purpose:** Balance the dataset and improve model generalization by creating synthetic examples for minority classes.
- **Techniques:** Synonym replacement, back-translation, paraphrasing, noise injection.
- **Output:** Augmented datasets, distribution plots, updated performance metrics.
- **Notes:** Always use augmented dataset for subsequent experiments to maintain consistency.

### 3. **Performance Benchmark**
- **Purpose:** Evaluate different base models (e.g., `roberta-base`, `deberta-v3-base`, `bertweet-base`) on stance classification.
- **Output:** Metrics per model (accuracy, precision, recall, F1), confusion matrices, comparison plots.
- **Notes:** Establishes the best-performing single model before ensembles or HPO.

### 4. **Model Ensembles**
- **Purpose:** Combine multiple models to improve F1-score beyond individual performance.
- **Techniques:** Hard voting, soft voting, weighted ensemble.
- **Output:** Ensemble predictions, metrics, confusion matrices, comparison with single models.
- **Notes:** Use the best single models from the performance benchmark; dataset should include data augmentation.

### 5. **Hyperparameter Optimization (HPO)**
- **Purpose:** Optimize hyperparameters of the best model (e.g., `roberta-base`) using Optuna.
- **Target Metric:** Macro F1-score on the validation set.
- **Parameters Optimized:** Learning rate, batch size, number of epochs, weight decay, warmup ratio, Adam betas.
- **Output:** Best hyperparameter configuration, optimized metrics, study visualizations, trained model.
- **Notes:** Requires GPU for efficient search; uses early stopping to avoid overfitting.

---

## Recommended Experiment Order: 

0. PEFT  
1. Performance Benchmark 
2. Data Augmentation  
3. Hyperparameter Optimization (HPO) 
4. Model Ensembles

## Notes

- All experiments should be reproducible using the provided notebooks.  
- Ensure GPU availability. 
- Always verify dataset consistency.
- Save outputs in the designated `experiments/text/` subfolders.
