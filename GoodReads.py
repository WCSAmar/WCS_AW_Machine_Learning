import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Load Dataset
#df = pd.read_csv('books.csv')
df = pd.read_csv(
    "books.csv",
    engine="python",
    on_bad_lines="skip"   #problem rows
)

df = df[['bookID', 'title', 'authors', 'average_rating', 'ratings_count']]

#Rename columns
df.columns = ['book_id', 'title', 'authors', 'rating', 'votes']
df = df.dropna()
df = df.reset_index(drop=True)
#=============================================================
#Popularity Recommender
def popularity_recommender(df, top_n=10):
       
    C = df['rating'].mean()
    
    #Minimum votes
    m = df['votes'].quantile(0.90)
    
    #Qualified books
    qualified = df[df['votes'] >= m].copy()
    
    #Weighted rating
    qualified['score'] = (
        (qualified['votes'] / (qualified['votes'] + m)) * qualified['rating'] +
        (m / (qualified['votes'] + m)) * C
    )
    
    qualified = qualified.sort_values('score', ascending=False)
    
    return qualified[['title', 'authors', 'rating', 'votes', 'score']].head(top_n)


#Missing authors
df['authors'] = df['authors'].fillna('')

#TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['authors'])

#Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#Create index mapping
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

#=======================================================================
def content_recommender(title, top_n=10):
    
    
    #Check if book exists
    if title not in indices:
        return "Book not found in dataset"
    
    #Get index
    idx = indices[title]
    
    #Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    #Sort books based on similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:top_n+1]
    
    # Get book indices
    book_indices = [i[0] for i in sim_scores]
    
    return df[['title', 'authors', 'rating']].iloc[book_indices]


if __name__ == "__main__":
    
    print("\nTop Popular Books:\n")
    print(popularity_recommender(df, top_n=10))
    
    print("\nBooks similar to 'Harry Potter and the Sorcerer's Stone':\n")
    print(content_recommender("Harry Potter and the Sorcerer's Stone", top_n=5))
