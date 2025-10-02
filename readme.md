# **Implicit Hate Speech Detection with BERT and Emotion Synthesis**

This project implements a deep learning model to detect implicit hate speech in text. This repository contains the code and resources for this by implementing a novel framework that leverages Supervised Contrastive Learning, attention mechanisms, and emotion synthesis to identify subtle and indirect forms of hate speech in text. It leverages a pre-trained BERT model for robust feature extraction and incorporates advanced data augmentation techniques to achieve state-of-the-art performance, surpassing existing benchmarks by 5-10% in Accuracy and F1 Score.

The core methodology and findings are detailed in our research paper: [Leveraging Supervised Contrastive Learning with Attention Mechanisms and Emotion Synthesis for Detecting Implicit Hate Speech](./paper/Implicit_Hate_Speech_Paper.pdf).

## **Project Structure**
```
Implicit-Hate-Speech-Detection 
│  
├── data/  
│   └── .gitkeep                             \# Directory for datasets (ignored by Git).  
│  
├── notebooks/  
│   └── main_analysis.ipynb                  \# Main Jupyter Notebook with the full analysis workflow.  
│  
├── paper/  
│   └── Implicit_Hate_Speech_Paper.pdf       \# The complete research paper for the project.  
│  
├── results/  
│   ├── class_distribution.                  \# Visualization of the dataset class distributions.  
│   ├── model_architecture.png               \# The system architecture diagram.  
│   └── final_metrics.txt                    \# A summary of the final model performance.
│   └── performance_comparison_ishate.png    \# Tabular Comparision & Analysis.  
│  
├── src/  
│   ├── __init__.py                          \# Makes 'src' a Python package.  
│   ├── data_processing.py                   \# Functions for loading and preparing data.  
│   ├── model.py                             \# The PyTorch model definition.  
│   └── train.py                             \# Functions for the training and evaluation loop.  
│  
├── README.md                                \# This overview file.  
└── requirements.txt                         \# Required Python libraries to run the project.
```

## **Technologies & Tools**

* **Primary Language:** Python  
* **Core Libraries:**  
  * **TensorFlow / Keras:** For building and training the deep learning model.  
  * **Hugging Face Transformers:** For using the pre-trained BERT model (bert-base-uncased).  
  * **Scikit-learn:** For data splitting and evaluation metrics.  
  * **Pandas:** For data manipulation.  
  * **NRCLex:** For emotion feature extraction.  
  * **PyTorch:** For model development and training.  
* **Development Environment:** Jupyter Notebook

## **Key Features & Methodology**

This project introduces a robust framework for detecting implicit hate speech by focusing on learning discriminative representations. The key components of the methodology are:

* **Large-Scale Data Augmentation:** Expanded the training dataset from \~20,000 to \~250,000 samples using techniques like paraphrasing, synonym replacement, and back-translation to improve model generalization.  
* **Advanced Feature Extraction:** Utilized **DeBERTa-v3-base** for powerful contextual text embeddings and integrated **NRCLex** to synthesize emotion and sentiment features, capturing affective cues in the text.  
* **Attention-Based Model Architecture:** Implemented a model with a word-level attention mechanism to focus on the most salient parts of the text for classification.  
* **Supervised Contrastive Learning:** The model was trained using a combination of Supervised Contrastive Loss and standard Cross-Entropy Loss. This encourages the model to pull representations of same-class samples closer together while pushing different-class samples apart, which is highly effective for distinguishing between nuanced categories like implicit hate and non-hate speech.

## **Results**

The proposed model demonstrates state-of-the-art performance, significantly outperforming standard baselines on the ISHate dataset.

* Overall Accuracy: 88.9%  
* Implicit Hate Speech F1-Score: 61.2% (a major improvement over standard models)

This confirms that the combination of data augmentation, emotion synthesis, and contrastive learning is a highly effective strategy for this challenging NLP task.

## 

## **How to Run**

1. **Clone the Repository:**  
   ```
    git clone [https://github.com/harshith1801/Implicit-Hate-Speech-Detection.git\](https://github.com/harshith1801/Implicit-Hate-Speech-Detection.git)  
   cd Implicit-Hate-Speech-Detection
   ```
2. **Set up a Virtual Environment (Recommended):**  
```
   python \-m venv venv  
   source venv/bin/activate    \# On Windows, use \`venv\\Scripts\\activate\`
```
3. **Install Dependencies:**  
```
   pip install \-r requirements.txt
```
4. **Add the Dataset:**

This repository does not include the data files. Please download the required datasets ```(cleaned_train_new_final.csv)``` and ```(implicit_test.csv)``` and place them inside the ```data``` folder. *You can find the datasets from the original source or a public repository where they are hosted.*

5. **Run the Jupyter Notebook:**  
   Launch Jupyter and open the main analysis notebook.
```
jupyter notebook notebooks/main_analysis.ipynb
```
You can then run the cells in the notebook to execute the full data processing, training, and evaluation pipeline.

## **Documentation**

A full, in-depth explanation of the methodology, experiments, and results is available in the **research paper** located in the `paper/` directory.

