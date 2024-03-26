# Load spacy and textblob in English:
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import matplotlib.pyplot as plt
# Load pandas to read csv
import pandas as pd

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Function 1: Sentiment scoring function with one input:
def sentiment(text):
    doc = nlp(text)
    popularity = round(doc._.blob.polarity, 3)            # Run Popularity query from textblob
    subjectivity = round(doc._.blob.subjectivity, 3)      # Run Subjectivity query from textblob
    return popularity, subjectivity
    # Output text display:
    #print(f'''The review '{text}' was analysed and given
    #Popularity score of: {round(popularity,3)}.
    #Subjectiviy score of: {round(subjectivity,3)}.''')

# Function 2: 
def sentiment_remove_stopwords(text):
    doc = nlp(text)
    # Step to remove stop words, multiple not function of punctuation, stop and spaces
    tokens = [token.orth_ for token in doc if not token.is_punct | token.is_stop | token.is_space]
    # Convert array of tokens back into sentence with join
    token_text = " ".join(tokens)
    # Put new sentence without stopwords through Function 1:
    return sentiment(token_text)

# Variance funtions used to plot differences between sentiment functions:
def pop_variance(text):
    # Index 0 used to return Popularity score
    return sentiment(text)[0] - sentiment_remove_stopwords(text)[0]

def sub_variance(text):
    # Index 1 used to return Subjectivity score
    return sentiment(text)[1] - sentiment_remove_stopwords(text)[1]

# Read the CSV file into a DataFrame
df = pd.read_csv('amazon_product_reviews.csv')

# Remove missing values for reviews.text from dataset
clean_df = df.dropna(subset=['reviews.text'])

# Number of reviews to be analysed = count. 
# Uncomment statement below and remove 100 value to allow for a user input
count = 100 #int(input("How many Amazon reviews would you like to sample this program on?"))

# Extract first {count} non blank reviews:
reviews_data = clean_df['reviews.text'][:count]

# Empty arrays used for charts
popularity_scores = []
subjectivity_scores = []

for review in reviews_data:
    # Pass text string of review through the four functions above
    standard = sentiment(review)
    cleaned = sentiment_remove_stopwords(review)
    pop_delta = pop_variance(review)
    sub_delta = sub_variance(review)
    popularity_scores.append(pop_delta)     # Used to create array of popularity variance scores
    subjectivity_scores.append(sub_delta)   # Used to create array of subjectivity variance scores

    # Output, only show first 200 characters of review in a text statement. Use index 0 for Popularity and 1 for Subjectivity.
    print(f"""Review: {review[:200]}... 
    Analysis with stop-words 
    Popularity: {standard[0]}, Subjectivity: {standard[1]}
    Analysis without stop-words 
    Popularity: {cleaned[0]}, Subjectivity: {cleaned[0]}
    Variance Popularity: {pop_delta},    Variance Subjectivity: {sub_delta}\n""")

# Create a chart to show the variance for popularity and subjectivity for the two NLP functions. 
plt.scatter(popularity_scores, subjectivity_scores)
plt.xlabel("Popularity delta: Sentiment Function less (No Stop Words) Sentiment Function")
plt.ylabel("Subjectivity delta: Sentiment Function less (No Stop Words) Sentiment Function")
plt.grid(True) 
plt.show()