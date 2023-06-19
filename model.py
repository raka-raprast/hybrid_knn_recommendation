import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import random


async def generate_recommendation(user_id, data):
    # Check if target_user_id exists in the dataset
    if user_id not in data['interacted_id'].unique():
        # Target user doesn't exist, recommend 10 random items
        recommended_post_ids_cf = random.sample(
            data['post_id'].unique().tolist(), k=10)
    else:
        # Create a user-item matrix
        user_item_matrix = pd.pivot_table(
            data, values='action', index='interacted_id', columns='post_id', aggfunc='count', fill_value=0)

        # Fit the Nearest Neighbors model
        k = 10  # Number of neighbors to consider
        model_cf = NearestNeighbors(metric='cosine', algorithm='brute')
        model_cf.fit(user_item_matrix.values)

        # Find nearest neighbors for target_user_id
        target_user_index = user_item_matrix.index.get_loc(user_id)
        distances, indices = model_cf.kneighbors(
            user_item_matrix.iloc[target_user_index].values.reshape(1, -1), n_neighbors=k+1)

        # Get recommended post_ids
        recommended_post_ids_cf = user_item_matrix.columns[indices.squeeze()].tolist()[
            1:]

        # Check if the number of recommended items is less than 10
        if len(recommended_post_ids_cf) < 10:
            # Add random items from existing post_ids to meet the required count
            remaining_count = 10 - len(recommended_post_ids_cf)
            random_post_ids = random.sample(
                data['post_id'].unique().tolist(), k=remaining_count)
            recommended_post_ids_cf += random_post_ids

    # Concatenate relevant post attributes into a single text column
    data['post_attributes'] = data['caption'] + ' ' + data['title'] + ' ' + \
        data['post_type'] + ' ' + \
        data['categories'].apply(lambda x: ' '.join(x))

    # Create a TF-IDF vectorizer and calculate the similarity matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['post_attributes'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Find similar posts based on the target_user_id's recent interactions
    recent_interactions = data[data['interacted_id'] == user_id]['post_id']
    recent_post_indices = data[data['post_id'].isin(recent_interactions)].index

    # Get similar post_ids
    similar_post_indices = np.argsort(similarity_matrix[recent_post_indices])[
        :, ::-1].flatten()
    recommended_post_ids_cb = data.iloc[similar_post_indices]['post_id'].unique(
    ).tolist()

    # Check if the number of recommended items is less than 10
    if len(recommended_post_ids_cb) < 10:
        # Add random items from existing post_ids to meet the required count
        remaining_count = 10 - len(recommended_post_ids_cb)
        random_post_ids = random.sample(
            data['post_id'].unique().tolist(), k=remaining_count)
        recommended_post_ids_cb += random_post_ids

    final_recommendations = recommended_post_ids_cf + \
        recommended_post_ids_cb[:10 - len(recommended_post_ids_cf)]

    return final_recommendations
