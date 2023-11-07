import os
import openai
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Assuming your functions from the previous code are defined in a file named `recommendation_functions.py`
from src.recommendation_functions.recommendation_functions import (
    filter_predictions,
    rerank_predictions,
)

# Load the datasets
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath("__file__")), "data")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
RAW_DATA_DIR = os.path.join(DATA_DIR, "goodreads_raw")

# Raw Data
books_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "books.csv"))
tags_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "tags.csv"))
book_tags_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "book_tags.csv"))

# Predictions Datasets
top_n_recommendations = pd.read_csv(
    os.path.join(PREDICTIONS_DIR, "top_n_recommendations.csv")
)
sorted_smaller_predictions_df = pd.read_csv(
    os.path.join(PREDICTIONS_DIR, "sorted_smaller_predictions.csv")
)


# OpenAI Creds
# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")


# Streamlit app
def main():
    st.title("Book Recommendation System")

    # Initialize or use existing session state variables
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
    if "user_request" not in st.session_state:
        st.session_state["user_request"] = ""
    if "show_recommendations" not in st.session_state:
        st.session_state["show_recommendations"] = False
    if "show_custom_recommendations" not in st.session_state:
        st.session_state["show_custom_recommendations"] = False

    # User ID selection
    user_ids = top_n_recommendations["user_id"].unique()
    st.session_state["user_id"] = st.selectbox(
        "Select a User ID:",
        user_ids,
        index=0
        if st.session_state["user_id"] is None
        else user_ids.tolist().index(st.session_state["user_id"]),
    )

    # Button to show top 10 recommendations
    if st.button("Show Top 10 Recommendations"):
        st.session_state["show_recommendations"] = True
        st.session_state["show_custom_recommendations"] = False

    # Call function to show top 10 recommendations if flag is True
    if st.session_state["show_recommendations"]:
        show_recommendations(st.session_state["user_id"])

    # Text input for custom recommendations request
    st.session_state["user_request"] = st.text_input(
        'Enter your request for refined book recommendations (e.g., "I want books about AI"):',
        st.session_state["user_request"],
    )

    # Button to get recommendations based on user request
    if st.button("Get Recommendations Based on Request"):
        st.session_state["show_custom_recommendations"] = True

    # Call function to show custom recommendations if flag is True
    if st.session_state["show_custom_recommendations"]:
        show_custom_recommendations(
            st.session_state["user_id"], st.session_state["user_request"]
        )


def show_recommendations(user_id):
    # Filter for user_id and sort by prediction score
    user_recommendations = top_n_recommendations[
        top_n_recommendations["user_id"] == user_id
    ].sort_values(by="prediction", ascending=False)
    # Merge with books_df to get book titles
    user_recommendations_with_titles = user_recommendations.merge(
        books_df[["book_id", "title"]], left_on="item_id", right_on="book_id"
    )
    # Display the top recommendations
    st.write(user_recommendations_with_titles)


def show_custom_recommendations(user_id, user_request):
    if user_request:  # Ensure there is a request to process
        # Call the filter_predictions function
        filtered_predictions = filter_predictions(
            user_id,
            sorted_smaller_predictions_df,
            books_df,
            tags_df,
            book_tags_df,
            user_request,
        )
        # Call the rerank_predictions function
        reranked_recommendations = rerank_predictions(
            filtered_predictions, books_df, user_request
        )
        # Display custom recommendations
        st.write(reranked_recommendations)
    else:
        st.write("Please enter a request to get custom recommendations.")


if __name__ == "__main__":
    main()
