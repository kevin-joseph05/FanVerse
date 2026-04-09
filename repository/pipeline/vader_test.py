from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline


def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)

    print(f"Sentiment Scores: {sentiment_dict}")
    print(f"Negative Sentiment: {sentiment_dict['neg']*100}%")
    print(f"Neutral Sentiment: {sentiment_dict['neu']*100}%")
    print(f"Positive Sentiment: {sentiment_dict['pos']*100}%")
    
    if sentiment_dict['compound'] >= 0.05:
        print("Overall Sentiment: Positive")
    elif sentiment_dict['compound'] <= -0.05:
        print("Overall Sentiment: Negative")
    else:
        print("Overall Sentiment: Neutral")


if __name__ == "__main__":

    print("\n1st Statement:")
    sentence = "Iam soo mad that my team lostttt"
    sentiment_scores(sentence)

    print("\n2nd Statement:")
    sentence = "Shweta played well in the match as usual."
    sentiment_scores(sentence)

    print("\n3rd Statement:")
    sentence = "Let's gooo real madrid won"
    sentiment_scores(sentence)