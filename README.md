# FakeNewsDetection![2](https://github.com/user-attachments/assets/5a8bb7ff-5208-4bac-aac1-47e9616d986e)
<img width="1914" height="1296" alt="lr-cr3" src="https://github.com/user-attachments/assets/975206a2-5794-4dc2-af18-396d3fc76000" />
# Fake News Detection Project

This repository contains Python code for building and evaluating machine learning models to detect fake news. The project implements a complete pipeline, from data loading and preprocessing to model training, evaluation, and interactive prediction.

## Project Overview

The goal of this project is to classify news articles as either "True News" or "Fake News" using various machine learning algorithms. The pipeline is designed to be clear, modular, and easy to run, especially within a Google Colaboratory environment.

The core pipeline follows these steps, as depicted in the project's architecture diagram:

1.  **Start & Dataset Loading**: Load the `Fake.csv` and `True.csv` datasets.
2.  **Preprocessing**: Clean and prepare the text data (e.g., lowercasing, removing punctuation, special characters, and numbers).
3.  **Feature Extraction**: Convert raw text into numerical features suitable for machine learning models using TF-IDF Vectorization.
4.  **Dataset Splitting**: Divide the dataset into Training and Testing sets.
5.  **Training the Classifier**: Train multiple classification models on the training data.
6.  **Model Evaluation**: Assess the performance of the trained models using metrics like accuracy and classification reports. Learning curves are also generated to understand model performance with varying training data sizes, which can inform "Model Tuning" (though explicit hyperparameter tuning is not implemented in this version, the curves provide insights).
7.  **Prediction**: Use the best-performing "Trained Model" to classify new, unseen "User Queries" as either "True News" or "Fake News".

## Features

* **Data Loading & Merging**: Reads `Fake.csv` and `True.csv` datasets and combines them.
* **Text Preprocessing**: Cleans news article text for model readiness.
* **TF-IDF Feature Extraction**: Converts text into numerical TF-IDF features.
* **Train/Test Split**: Divides data into training and testing sets for robust evaluation.
* **Multiple Classifier Support**: Evaluates several popular machine learning classifiers:
    * Random Forest Classifier
    * Logistic Regression
    * Gradient Boosting Classifier
    * Decision Tree Classifier
* **Comprehensive Evaluation**: Provides accuracy scores and detailed classification reports for each model.
* **Learning Curve Visualization**: Generates and saves learning curves to assess model bias/variance.
* **Interactive Manual Testing**: Allows users to input custom news text and get real-time predictions from the trained models.
* **Google Colab Ready**: Optimized for easy setup and execution in Google Colaboratory.

## Setup and Installation (Google Colab)

This project is designed to be run in Google Colaboratory. Follow these steps to set it up:

1.  **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com/) and create a new notebook.

2.  **Upload Datasets to Google Drive**:
    * Create a folder in your Google Drive (e.g., `My Drive/FakeNewsProject/data/`).
    * Upload your `Fake.csv` and `True.csv` files into this folder.
    * You can download these datasets from Kaggle (e.g., [Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-news-detection)).

3.  **Mount Google Drive (Colab Cell 1)**:
    * In your Colab notebook, add a new code cell and paste the following:
        ```python
        from google.colab import drive
        drive.mount('/content/drive')
        ```
    * Run this cell. It will prompt you to authorize Google Drive access. Follow the instructions.

4.  **Copy `run_fake_news_detection_pipeline` Code (Colab Cell 2)**:
    * Copy the entire code from the `run_fake_news_detection_pipeline` function (and all helper functions like `get_news_label`, `preprocess_text`, `process_user_query_and_predict`, `train_and_evaluate_classifier`, `plot_model_evaluation_curve`) into a *new* code cell in your Colab notebook.

5.  **Configure Data Paths and Run Pipeline (Colab Cell 3)**:
    * In a *new* code cell, define your data and output paths and call the main pipeline function:
        ```python
        # IMPORTANT: Set this to the actual path where your Fake.csv and True.csv are located in Google Drive.
        # Example: If your files are in 'My Drive/my_project/data/', set it to '/content/drive/My Drive/my_project/data'
        data_input_path = '/content/drive/My Drive/FakeNewsProject/data'
        output_destination_path = '/content/drive/My Drive/FakeNewsProject/output' # This folder will be created if it doesn't exist

        run_fake_news_detection_pipeline(data_input_path, output_destination_path, test_data_split_ratio=0.2, random_seed=42)
        ```
    * **Make sure to update `data_input_path` to your specific Google Drive folder.**
    * Run this cell.

## Usage

Once the setup is complete and you run the cells sequentially in Google Colab:

1.  The script will load and preprocess the data.
2.  It will then train and evaluate each specified classifier.
3.  Evaluation results (accuracy, classification reports) will be printed to the console and saved to a text file in your specified output directory (`/content/drive/My Drive/FakeNewsProject/output/results/`).
4.  Learning curve plots for each model will be generated and saved as PNG images in the same output directory.
5.  For each trained model, the script will prompt you to `Enter news text to test...`. You can type or paste a news article, and the model will predict whether it's "Fake News" or "True News".

## Models Used

The project evaluates the following scikit-learn classifiers:

* `RandomForestClassifier`
* `LogisticRegression`
* `GradientBoostingClassifier`
* `DecisionTreeClassifier`

## Output

The script generates:

* **Console Output**: Detailed information about data loading, preprocessing steps, model training progress, and evaluation metrics.
* **`model_evaluation_report_YYYYMMDD_HHMMSS.txt`**: A text file in the `output/results/` directory containing the accuracy score and classification report for each model.
* **`learning_curve_MODEL_NAME.png`**: PNG image files in the `output/results/` directory, visualizing the learning curve for each trained model.
* **`manual_testing_data.csv`**: A CSV file containing the 20 articles (10 fake, 10 true) separated for manual testing purposes.

## Contributing

Feel free to fork this repository, open issues, or submit pull requests. Contributions are welcome!

## License

This project is open-source and available under the [MIT License](LICENSE).
