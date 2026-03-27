import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# ---- LOAD MODELS ----
sid = SentimentIntensityAnalyzer()
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)
# ---- LOAD DATA ----
# update this path to wherever Rainier's JSON file lives
with open("reddit.json", "r") as f:
    posts = json.load(f)

# ---- KEYWORD RULES FOR BEHAVIORAL PATHWAY ----
PATHWAY_KEYWORDS = {
    "loyalty_signal": ["ride or die", "been a fan since", "never switching", "my team", "always supported", "love this team", "season tickets"],
    "churn_risk": ["done", "last straw", "moving on", "can't support", "disappointed", "losing faith", "hate", "joke"],
    "conversion_trigger": ["just started watching", "first game", "got me into", "new fan", "hooked", "converted me"],
    "community_influence": ["who's going", "watch party", "outfit check", "let's go", "game day", "who else"],
    "purchase_intent": ["buying", "merch", "tickets", "just ordered", "where can I get", "season pass", "reselling"],
    "identity_attachment": ["my rook", "our team", "she's the reason", "inspires me", "means everything", "don't wanna see her go"],
    "disengagement_marker": ["stopped watching", "don't care anymore", "used to watch", "lost interest", "not worth it"]
}

# ---- KEYWORD RULES FOR PRIORITY SIGNAL ----
PRIORITY_KEYWORDS = {
    "loyalty_stress": ["losing", "scandal", "trade", "cut", "protect", "leave", "losing faith", "last straw"],
    "identity_anchor": ["she's the reason", "follow her", "wherever she goes", "my player", "rook", "protect"],
    "conversion_moment": ["first game", "got me into", "started watching", "hooked", "new fan"],
    "cross_sport_superfan": ["also watch", "both leagues", "NWSL and WNBA", "love both", "multi-sport"],
    "trust_split": ["love the players", "hate the organization", "front office", "management", "ownership"]
}

def classify_pathway(text):
    text_lower = text.lower()
    scores = {}
    for pathway, keywords in PATHWAY_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[pathway] = count
    if scores:
        return max(scores, key=scores.get)
    return "none"

def classify_priority(text):
    text_lower = text.lower()
    scores = {}
    for signal, keywords in PRIORITY_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[signal] = count
    if scores:
        return max(scores, key=scores.get)
    return "none"

# ---- MAIN PIPELINE ----
results = []

for post in posts:
    text = post["text"]

    # VADER Sentiment
    vader = sid.polarity_scores(text)
    compound = vader["compound"]
    sentiment_score = round((compound + 1) / 2, 2)

    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Hugging Face emotion
    emotions = emotion_classifier(text)[0]
    top_emotion = max(emotions, key=lambda x: x["score"])
    emotional_affinity_score = round(top_emotion["score"] * 100)

    # Behavioral pathway
    behavioral_pathway = classify_pathway(text)

    # Priority signal
    priority_signal = classify_priority(text)  

    # Confidence score (average of VADER confidence + emotion confidence)
    vader_confidence = abs(compound)
    emotion_confidence = top_emotion["score"]
    confidence_score = round((vader_confidence + emotion_confidence) / 2, 2)

    results.append({
        "post_id": post["post_id"],
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "emotional_affinity_score": emotional_affinity_score,
        "behavioral_pathway": behavioral_pathway,
        "confidence_score": confidence_score,
        "priority_signal": priority_signal
    })

# ---- SAVE OUTPUT ----
with open("signals_reddit_output.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Processed {len(results)} posts")
print(f"Saved to signals_output.json")

# ---- QUICK SUMMARY ----
for r in results[:5]:
    print(f"\nPost: {r['post_id'][:8]}...")
    print(f"  Sentiment: {r['sentiment']} ({r['sentiment_score']})")
    print(f"  Affinity: {r['emotional_affinity_score']}")
    print(f"  Pathway: {r['behavioral_pathway']}")
    print(f"  Signal: {r['priority_signal']}")