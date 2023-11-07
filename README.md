# Neural Collaborative Filtering for Book Recommendations

This repository contains the implementation of a neural collaborative filtering (NCF) system for book recommendations, using explicit and implicit feedback from users.

## Overview

The system utilizes a deep learning architecture to predict user preferences for books based on past interaction data. The model combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) to learn from user-book interactions, creating a hybrid recommendation system that captures both low-level interactions and high-level abstractions.

## Repository Structure

```
root/
 ├── dev-data/                    # Directory for datasets and predictions
 │   ├── data/                    # Folder containing the raw datasets
 │   │   ├── ratings.csv          # User-book ratings
 │   │   ├── to_read.csv          # Books marked 'to read' by users
 │   │   ├── books.csv            # Book metadata
 │   │   ├── book_tags.csv        # Tags associated with books
 │   │   └── tags.csv             # Tag metadata
 │   └── predictions/             # Folder for prediction outputs
 ├── .env                         # Environment variables for API keys
 └── recommendation_script.py     # Main script for the recommendation system
```

## Datasets

The datasets include:
- `ratings.csv`: User-book ratings.
- `to_read.csv`: Books marked 'to read' by users.
- `books.csv`: Book metadata including authors, title, and average ratings.
- `book_tags.csv`: Tags associated with books.
- `tags.csv`: Tag metadata.

## Setup

1. Clone the repository to your local machine.
2. Ensure that Python 3.x is installed.
3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Place your datasets in the `dev-data/data` directory.
5. Set your OpenAI API key in a `.env` file.

## Usage

Run the `recommendation_script.py` script to process the data and generate recommendations:
```
python recommendation_script.py
```

## Model Training

The training process includes:
- Loading and preprocessing data.
- Normalizing ratings and combining explicit and implicit feedback.
- Balancing the dataset by resampling.
- Defining and training the NCF model with early stopping.
- Saving model predictions.

## Recommendations Generation

The script includes functions to:
- Generate batch predictions.
- Rank and filter recommendations based on user requests.
- Rerank recommendations according to user preferences using OpenAI's GPT-4 model.

## Evaluation

The system's performance can be evaluated using the provided metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

## Environment Variables

You must provide your OpenAI API key in a `.env` file for the reranking function to work.

## Contributions

Contributions to this project are welcome. Please ensure that you follow the existing coding style and add unit tests for any new or changed functionality.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

---

For a detailed explanation of each component of the code, refer to the inline comments within the `recommendation_script.py` script.
