import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect

# Download NLTK resources (if not already downloaded)
nltk.download('vader_lexicon')



def detect_language(text):
    return detect(text)




def load_spacy_model(language):
    # Load spaCy model based on detected language
    if language == 'en':
        return spacy.load("en_core_web_sm")
    # Add more languages as needed
    else:
        print("Language not supported, defaulting to English.")
        return spacy.load("en_core_web_sm")  # Default to English if language is not supported


def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score

def extract_keywords(text, nlp):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']]

def recognize_entities(text, nlp):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def main():
    text = input("Enter the text to analyze: ")

    # Detect language of the input text
    language = detect_language(text)
    print("Detected Language:", language)

    # Load spaCy model based on detected language
    nlp = load_spacy_model(language)

    sentiment_score = analyze_sentiment(text)
    print("\nSentiment Analysis Results:")
    print("Positive: {:.2f}%".format(sentiment_score['pos']*100))
    print("Neutral: {:.2f}%".format(sentiment_score['neu']*100))
    print("Negative: {:.2f}%".format(sentiment_score['neg']*100))
    print("Compound Score: {:.2f}".format(sentiment_score['compound']))

    keywords = extract_keywords(text, nlp)
    print("\nKeywords:")
    print(keywords)

    entities = recognize_entities(text, nlp)
    print("\nNamed Entities:")
    print(entities)

if __name__ == "__main__":
    main()
