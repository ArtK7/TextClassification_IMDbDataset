## Sentiment Analysis Using Fine-Tuned DistilBERT

### Overview

This project implements sentiment analysis on the IMDb movie reviews dataset using a fine-tuned DistilBERT model.
Sentiment analysis is a key task in Natural Language Processing (NLP) that involves classifying text into positive or negative sentiments. 
The goal is to leverage a pre-trained, fine-tuned DistilBERT model to achieve efficient and accurate sentiment classification. The project also incorporates preprocessing, tokenization, and dynamic balancing to handle class imbalances effectively, while evaluation metrics such as accuracy, precision, recall, and F1 score are used to measure performance.
### Key Features

* Model: DistilBERT, a lightweight and efficient transformer model, fine-tuned on the IMDb dataset.
* Dataset: IMDb dataset consisting of 50,000 labeled movie reviews (positive and negative sentiments).
* Dynamic Balancing: Ensures proportional representation of classes during evaluation.
* Metrics: Evaluation using accuracy, precision, recall, and F1 score.
* Preprocessing: Includes tokenization, truncation, and padding to prepare text data for the model.
* Performance: Exceeds 85% across all evaluation metrics.

### Workflow

1. Dataset Preparation: The IMDb dataset is loaded using the HuggingFace datasets library. Text data is preprocessed using the DistilBERT tokenizer, including truncation and padding.
2. Dynamic Balancing: The test dataset is shuffled, and a balanced subset (250 samples) is created with an even distribution of positive and negative reviews.
3. Model Selection: A pre-trained DistilBERT model fine-tuned on IMDb is used for classification.
4. Prediction and Evaluation: The model generates predictions for the balanced test dataset. Metrics such as accuracy, precision, recall, and F1 score are calculated and analyzed.
5. Result Interpretation: Results are validated against benchmarks to demonstrate the model’s performance and reliability.
### Setup

#### Prerequisites

Ensure the following are installed on your system:
*     Python (>= 3.8)
*     pip (Python package manager)
#### Running the Script

Clone this repository or copy the prototype.py file to your local machine.

Execute the script:
*     python prototype.py

Install the required libraries:
These dependencies can be installed with the following command:

*     pip install transformers datasets torch scikit-learn colorama tqdm


### Evaluation Metrics

The model’s performance is evaluated using the following metrics:
* Accuracy: Overall correctness of predictions.
* Precision: Ratio of true positive predictions to all positive predictions.
* Recall: Ratio of true positive predictions to all actual positives.
* F1 Score: Harmonic mean of precision and recall.
#### Sample Results
* Metric    Value
* Accuracy:	0.87
* Precision:	0.89
* Recall:	0.85
* F1 Score:	0.87

### Project Highlights

* **Why DistilBERT?** Lightweight and efficient, DistilBERT achieves 97% of BERT’s performance while being 60% faster and using 40% fewer resources. The model’s fine-tuning for IMDb ensures domain-specific accuracy.
* **Dynamic Balancing:** Real-world datasets often suffer from class imbalances. This project uses a dynamic approach to balance positive and negative samples during evaluation.

### Future Improvements

* Extend sentiment analysis to multi-class problems (e.g., neutral sentiment).
* Incorporate additional datasets to evaluate generalization capabilities.
* Explore alternative transformer models such as Longformer for longer text inputs.

### Acknowledgments

* HuggingFace transformers library for pre-trained models and tokenizers.
* IMDb dataset for providing a benchmark corpus for sentiment analysis.
* Python libraries such as torch and scikit-learn for machine learning tools.

### Contact

For any questions or suggestions, feel free to reach out:
* Name: Art Kastrati
* Email: artkastrati523@gmail.com
* LinkedIn: www.linkedin.com/in/artkastrati777
