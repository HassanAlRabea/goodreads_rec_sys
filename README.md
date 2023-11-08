# Book Recommendation System

## Overview
This repository contains the implementation of a book recommendation system that leverages the power of Neural Collaborative Filtering (NCF) and GPT-4 from OpenAI for personalized book recommendations.

## Features
- Data normalization and processing for user and book IDs.
- Implementation of NeuMF, a fusion of Matrix Factorization (MF) and Multi-Layer Perceptron (MLP) for recommendation.
- Batch prediction with incremental saving and crash handling.
- Dynamic sampling of prediction results to reduce file size.
- User-specific book recommendations with reranking based on textual requests using GPT-4.

## OpenAI GPT-4 Integration
The system uses OpenAI's GPT-4 for understanding and processing user textual requests to filter and rerank recommendations. The interaction with GPT-4 is handled through the OpenAI API, which requires an API key for access.

## Dataset
The dataset includes user ratings, to-read lists, book metadata, and book tags. These are located in the `data/goodreads_raw` directory and should be structured as follows:
- `ratings.csv`: User-book interactions with explicit ratings.
- `to_read.csv`: User-book interactions with implicit feedback.
- `books.csv`: Metadata of books.
- `book_tags.csv`: Tags associated with books.
- `tags.csv`: Tag information.

## Prerequisites
- Python 3.x
- Pandas library
- NumPy library
- TensorFlow 2.x
- Keras
- scikit-learn
- Keras-tuner
- dotenv
- OpenAI (for GPT-4 API access)

Ensure that all the dependencies are installed using the following command:
```sh
pip install -r requirements.txt
```

## Installation
Clone the repository to your local machine:
```sh
git clone [repository link]
```

## Environment Variables
This project utilizes a `.env` file to store sensitive information such as API keys. Make sure to create a `.env` file in the root of the project and add the following line:
```
OPENAI_API_KEY='Your OpenAI API Key Here'
```

## Usage
Run the main script to launch the application:
```sh
python app.py
```

## Model Training
The NeuMF model architecture is defined in the script. You can train the model using the provided data by running the training block in the script. Or leverage the provided model file.

## Batch Predictions
After training, you can perform batch predictions and save them incrementally. The script `batch_predict_and_save` handles the prediction and saving process.

## Recommendation Generation
The system generates top N recommendations for each user, which can be filtered and reranked based on textual user requests. Examples are provided in the script.

## Customization
You can adjust the model parameters, batch sizes, and other configurations as per your system's capabilities and requirements.
