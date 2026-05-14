from flask import Flask, request, jsonify, render_template
import pickle, re, os
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

MODEL_DIR = r'D:\Amazon\models'

# 初始化 NLTK
for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find('tokenizers/' + resource)
    except LookupError:
        try:
            nltk.data.find('corpora/' + resource)
        except LookupError:
            nltk.download(resource, quiet=True)

# 加载模型
print('Loading model...')
with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'handcraft_features.pkl'), 'rb') as f:
    handcraft_features = pickle.load(f)

STOP_WORDS = set(stopwords.words('english'))
NEGATION_WORDS = {'no', 'not', 'never', 'neither', 'nor', 'nothing', 'nobody', 'nowhere', 'hardly', 'barely', 'scarcely'}
STOP_WORDS = STOP_WORDS - NEGATION_WORDS
LEMMATIZER = WordNetLemmatizer()
VADER = SentimentIntensityAnalyzer()

LABEL_NAMES = {0: '1 Star', 1: '2 Stars', 2: '3 Stars', 3: '4 Stars', 4: '5 Stars'}
STAR_COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

print('Model loaded. Ready!')

def preprocess_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'\b(not|no|never|neither|nor|hardly|barely|scarcely)\s+(\w+)', r'\1_\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(review_text):
    clean_text = preprocess_text(review_text)
    tfidf_feat = tfidf_vectorizer.transform([clean_text])

    char_length = len(clean_text)
    word_count = len(clean_text.split())
    exc_density = clean_text.count('!') / (len(clean_text) + 1)
    ques_density = clean_text.count('?') / (len(clean_text) + 1)
    upper_count = sum(1 for c in str(review_text) if c.isupper())
    vader = VADER.polarity_scores(str(review_text))

    hand_feat = np.array([[char_length, word_count, exc_density, ques_density, upper_count,
                            vader['neg'], vader['neu'], vader['pos'], vader['compound']]], dtype=np.float64)
    hand_feat_scaled = scaler.transform(hand_feat)

    combined = hstack([tfidf_feat, csr_matrix(hand_feat_scaled)])
    return combined, {
        'clean_text': clean_text,
        'char_length': char_length,
        'word_count': word_count,
        'vader_neg': round(vader['neg'], 4),
        'vader_neu': round(vader['neu'], 4),
        'vader_pos': round(vader['pos'], 4),
        'vader_compound': round(vader['compound'], 4),
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review_text = data.get('review', '').strip()

    if not review_text:
        return jsonify({'error': 'Please enter a review text'}), 400
    if len(review_text) < 10:
        return jsonify({'error': 'Review text is too short (minimum 10 characters)'}), 400

    features, info = extract_features(review_text)

    # 预测概率
    proba = model.predict_proba(features)[0]  # shape: (5,)
    predicted_label = int(model.predict(features)[0])

    # 构建每个星级的概率
    star_probs = []
    for i in range(5):
        star_probs.append({
            'star': i + 1,
            'label': LABEL_NAMES[i],
            'probability': round(float(proba[i]) * 100, 2),
            'color': STAR_COLORS[i],
        })
    star_probs.sort(key=lambda x: x['probability'], reverse=True)

    # 情感解释
    compound = info['vader_compound']
    if compound >= 0.5:
        sentiment = 'Very Positive'
    elif compound >= 0.05:
        sentiment = 'Positive'
    elif compound > -0.05:
        sentiment = 'Neutral'
    elif compound > -0.5:
        sentiment = 'Negative'
    else:
        sentiment = 'Very Negative'

    result = {
        'predicted_star': int(predicted_label + 1),
        'predicted_label': LABEL_NAMES[predicted_label],
        'confidence': star_probs[0]['probability'],
        'star_probs': star_probs,
        'sentiment': sentiment,
        'vader_compound': info['vader_compound'],
        'vader_pos': info['vader_pos'],
        'vader_neg': info['vader_neg'],
        'vader_neu': info['vader_neu'],
        'char_length': info['char_length'],
        'word_count': info['word_count'],
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5002, debug=False)
