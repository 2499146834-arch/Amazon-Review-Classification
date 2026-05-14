import os, sys, warnings, re, pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SEED = 42
np.random.seed(SEED)

for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find('tokenizers/' + resource)
    except LookupError:
        try:
            nltk.data.find('corpora/' + resource)
        except LookupError:
            nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words('english'))
NEGATION_WORDS = {'no', 'not', 'never', 'neither', 'nor', 'nothing', 'nobody', 'nowhere', 'hardly', 'barely', 'scarcely'}
STOP_WORDS = STOP_WORDS - NEGATION_WORDS
LEMMATIZER = WordNetLemmatizer()
VADER = SentimentIntensityAnalyzer()

MODEL_DIR = r'D:\Amazon\model'
os.makedirs(MODEL_DIR, exist_ok=True)
OUTPUT_DIR = r'D:\Amazon\experiment_outputs'

print('Loading data...')
DATA_LOADED = False
records = []

import requests, io
url = 'https://raw.githubusercontent.com/aman17292/Amazon-Reviews-Sentiment-Analysis/master/amazon_reviews.csv'
try:
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        df_temp = pd.read_csv(io.StringIO(resp.text))
        text_col, star_col = None, None
        for col in df_temp.columns:
            col_lower = col.lower()
            if text_col is None and any(k in col_lower for k in ['review', 'text', 'body', 'comment', 'summary']):
                text_col = col
            if star_col is None and any(k in col_lower for k in ['star', 'rating', 'score', 'class', 'sentiment']):
                star_col = col
        if text_col and star_col:
            for _, row in df_temp.iterrows():
                records.append({'reviewText': str(row[text_col]), 'star_rating': int(float(row[star_col]))})
            DATA_LOADED = True
            print('  Loaded ' + str(len(records)) + ' real reviews from URL')
except Exception:
    pass

if not DATA_LOADED:
    print('  URL unavailable, using synthetic data...')
    np.random.seed(SEED)
    pos_openings = ["I absolutely love this", "Really happy with my purchase", "This is hands down the best", "Excellent quality", "Could not be happier with this", "This product is fantastic", "Incredible value for money", "I highly recommend this", "The build quality is outstanding", "Worth every penny", "Super impressed with the quality", "Really well made", "I rarely write reviews but this deserves one", "Great performance", "This is a game changer", "Absolutely brilliant", "Better than the name brand alternatives", "Cannot fault this at all", "I use this daily and it has never let me down"]
    pos_bodies = ["Setup was incredibly easy, took only a few minutes.", "Works seamlessly with all my other devices.", "Arrived well-packaged and in perfect condition.", "The design is sleek and modern.", "Battery life has been impressive so far.", "The instructions were clear and easy to follow.", "Shipping was fast, got it two days earlier than expected.", "Build quality feels premium and durable.", "Performance has been consistently excellent.", "Really intuitive interface, no learning curve at all.", "The size is perfect for what I needed.", "Compatible with everything I've tried so far.", "Does exactly what it claims to do without any issues.", "Lightweight but feels solid in hand.", "Much better than the previous version I had."]
    pos_endings = ["Would definitely buy again!", "Highly recommended.", "You will not regret this purchase.", "Best purchase this year.", "Solid 5 stars, no question.", "Exactly what I was looking for."]
    neu_openings = ["This product is decent but", "It's an okay purchase, however", "Not bad, but not great either.", "Average product overall.", "This does the job but", "It's a mixed bag with this one.", "I have some mixed feelings about this", "Serviceable but", "It's fine for the price I paid, though", "Nothing special"]
    neu_bodies = ["It's functional but lacks some features I was hoping for.", "The quality is acceptable for the price point.", "Does what it says but nothing more than that.", "Worked fine for about a month, then started having minor issues.", "The design is okay but feels a bit cheap in hand.", "Setup was a bit confusing but eventually got it working.", "Performance is inconsistent -- sometimes great, sometimes not.", "It's not as durable as I expected from the description.", "Good enough to keep but probably would not buy again."]
    neu_endings = ["Three stars seems fair.", "An acceptable product at this price.", "Could be better, could be worse.", "Middle of the road."]
    neg_openings = ["Really disappointed with this", "This product is terrible", "I regret buying this", "Complete waste of money", "Did not work as advertised at all", "Very poor quality", "I had high hopes but this failed completely", "Save your money", "The worst purchase I've made", "Absolutely useless", "Shocked at how bad this is", "This broke within the first week", "Do not buy this", "Horrible experience", "This product is defective", "One star is too generous", "Nothing but problems since day one", "Utterly disappointed", "What a piece of junk", "Not fit for purpose", "Stopped working after just a few uses"]
    neg_bodies = ["The materials feel incredibly cheap and flimsy.", "Came with scratches and looked used already.", "Does not fit properly even though the listing said it would.", "Customer service was completely unhelpful.", "The battery died completely after two weeks.", "It overheats after 10 minutes of use.", "Instructions were impossible to follow.", "The app required to use it crashes constantly.", "Made a loud buzzing noise right out of the box.", "The color is completely different from the photos.", "Arrived three weeks late and the box was crushed.", "Requires a subscription not mentioned in the listing.", "The charging port was loose and barely makes contact.", "Advertised as waterproof but got damaged from light rain.", "Tried to get a refund and the seller refused.", "Plastic parts started cracking after a month.", "It is much smaller than it appears in the photos.", "The adhesive barely holds and it fell off within a day."]
    neg_endings = ["Stay away from this product.", "Wish I could give zero stars.", "Do yourself a favor and skip this one.", "Absolutely disgraceful quality control."]

    def generate_review(rating):
        if rating >= 4:
            o, b, e = np.random.choice(pos_openings), np.random.choice(pos_bodies), np.random.choice(pos_endings)
        elif rating == 3:
            o, b, e = np.random.choice(neu_openings), np.random.choice(neu_bodies), np.random.choice(neu_endings)
        else:
            o, b, e = np.random.choice(neg_openings), np.random.choice(neg_bodies), np.random.choice(neg_endings)
        if np.random.random() < 0.5:
            return o + ". " + b
        else:
            return o + ". " + b + " " + e

    n_samples = 6000
    rating_dist = [0.08, 0.05, 0.08, 0.19, 0.60]
    all_ratings = []
    for rating_val, p in enumerate(rating_dist, start=1):
        count = int(round(n_samples * p))
        all_ratings.extend([rating_val] * count)
    while len(all_ratings) < n_samples:
        all_ratings.append(np.random.choice([1, 2, 3, 4, 5], p=rating_dist))
    all_ratings = all_ratings[:n_samples]
    np.random.shuffle(all_ratings)
    used_hashes = set()
    for rating in all_ratings:
        for _ in range(10):
            review = generate_review(rating)
            rh = hash(review[:50])
            if rh not in used_hashes:
                used_hashes.add(rh)
                records.append({'reviewText': review, 'star_rating': rating})
                break

df_raw = pd.DataFrame(records)
df_raw = df_raw[df_raw['reviewText'].str.len() > 10].reset_index(drop=True)
print('  Total: ' + str(len(df_raw)) + ' reviews')

print('Preprocessing...')
def preprocess_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'\b(not|no|never|neither|nor|hardly|barely|scarcely)\s+(\w+)', r'\1_\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_clean(text):
    tokens = nltk.word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 1]
    return tokens

df_raw['cleanText'] = df_raw['reviewText'].apply(preprocess_text)
df_raw['tokens'] = df_raw['cleanText'].apply(tokenize_and_clean)
df_raw['char_length'] = df_raw['cleanText'].str.len()
df_raw['word_count_feat'] = df_raw['cleanText'].str.split().str.len()
df_raw['exclamation_density'] = df_raw['cleanText'].str.count('!') / (df_raw['cleanText'].str.len() + 1)
df_raw['question_density'] = df_raw['cleanText'].str.count('\\?') / (df_raw['cleanText'].str.len() + 1)
df_raw['upper_count'] = df_raw['reviewText'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
vader_scores = df_raw['reviewText'].apply(lambda x: VADER.polarity_scores(str(x)))
df_raw['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
df_raw['vader_neu'] = vader_scores.apply(lambda x: x['neu'])
df_raw['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
df_raw['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
df_raw['label'] = df_raw['star_rating'] - 1

print('Feature engineering...')
tfidf_vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=5, max_df=0.8, sublinear_tf=True, strip_accents='unicode', token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z_]+\b')
X_tfidf_all = tfidf_vectorizer.fit_transform(df_raw['cleanText'])
handcraft_features = ['char_length', 'word_count_feat', 'exclamation_density', 'question_density', 'upper_count', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
X_handcraft = df_raw[handcraft_features].values.astype(np.float64)
scaler = StandardScaler()
X_handcraft_scaled = scaler.fit_transform(X_handcraft)
X_combined = hstack([X_tfidf_all, csr_matrix(X_handcraft_scaled)])
y_all = df_raw['label'].values

print('Training best model (LR + TF-IDF tuned)...')
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_all, test_size=0.15, stratify=y_all, random_state=SEED)
best_model = LogisticRegression(C=3.0, max_iter=2000, class_weight='balanced', random_state=SEED)
best_model.fit(X_train, y_train)
print('  Train Accuracy: ' + str(round(best_model.score(X_train, y_train), 4)))
print('  Test Accuracy: ' + str(round(best_model.score(X_test, y_test), 4)))

print('Saving model...')
files = {'tfidf_vectorizer': tfidf_vectorizer, 'scaler': scaler, 'model': best_model, 'handcraft_features': handcraft_features}
for name, obj in files.items():
    path = os.path.join(MODEL_DIR, name + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print('  Saved: ' + path)
print('Done! Model saved to ' + MODEL_DIR)
