import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK resources (if not already downloaded)
nltk.download('vader_lexicon')

# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score

def extract_keywords(text):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']]

def recognize_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def main():
    text = input("Enter the text to analyze: ")
    
    sentiment_score = analyze_sentiment(text)
    print("\nSentiment Analysis Results:")
    print("Positive: {:.2f}%".format(sentiment_score['pos']*100))
    print("Neutral: {:.2f}%".format(sentiment_score['neu']*100))
    print("Negative: {:.2f}%".format(sentiment_score['neg']*100))
    print("Compound Score: {:.2f}".format(sentiment_score['compound']))

    keywords = extract_keywords(text)
    print("\nKeywords:")
    print(keywords)

    entities = recognize_entities(text)
    print("\nNamed Entities:")
    print(entities)

if __name__ == "__main__":
    main()