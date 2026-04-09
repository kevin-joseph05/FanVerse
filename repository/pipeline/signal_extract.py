import json
import sys
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from langdetect import detect, detect_langs, LangDetectException
from deep_translator import GoogleTranslator

# ---- CLIENT CONFIGURATION ----
if len(sys.argv) < 2:
    print("Usage: python3 signal_extract.py <client_name>")
    print("Example: python3 signal_extract.py 5wins")
    print("         python3 signal_extract.py allez_sports")
    sys.exit(1)

CLIENT = sys.argv[1]
REPO_PATH = Path(__file__).parent.parent / "data" / "clients" / f"{CLIENT}_repository.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "output" / f"repository_signals_{CLIENT}.json"

if not REPO_PATH.exists():
    print(f"ERROR: {REPO_PATH} not found")
    sys.exit(1)

# ---- LOAD MODELS ----
sid = SentimentIntensityAnalyzer()
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    truncation=True,
    max_length=512,
)

# ---- LOAD DATA ----
print(f"Loading {CLIENT} data from {REPO_PATH}...")
posts = []
with open(REPO_PATH, "r") as f:
    for line in f:
        posts.append(json.loads(line))

# ---- KEYWORD RULES FOR BEHAVIORAL PATHWAY ----
# Instagram-optimized phrase matching (short, enthusiastic language)
PATHWAY_KEYWORDS = {
    "loyalty_signal": ["lets go", "hypeddd", "yassss", "get the gold", "great to see", "we will"],
    "churn_risk": ["i'm done with", "done watching", "not watching anymore", "lost me as a fan", "can't support them anymore", "last straw", "unsubscribing", "unfollowing", "moving on", "gave up on", "hard to keep supporting", "lost me as"],
    "conversion_trigger": ["best [thing] i ever", "just watched", "just saw", "first time", "never knew", "finally", "discovered", "can't believe"],
    "community_influence": ["how do i join", "show up", "let's go @", "meet up", "who's in", "sign up", "get ready to"],
    "purchase_intent": ["just ordered", "where can i get", "want to buy", "buying tickets", "season pass"],
    "identity_attachment": ["love", "you rock", "so proud", "i remember", "best", "way to go", "you deserve", "learned from", "great seeing"],
    "disengagement_marker": ["stopped watching", "don't care anymore", "used to watch", "lost interest", "not worth it"]
}

# ---- KEYWORD RULES FOR PRIORITY SIGNAL ----
# Instagram-optimized phrase matching (short, enthusiastic language)
PRIORITY_KEYWORDS = {
    "loyalty_stress": ["will miss", "miss you", "come back", "don't go"],
    "identity_anchor": ["she's the reason", "follow her", "because of her", "you rock"],
    "conversion_moment": ["just watched", "just saw", "first time", "discovered", "best [thing] i ever"],
    "cross_sport_superfan": ["also watch", "both", "multiple"],
    "trust_split": ["love the", "hate the"]
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

def detect_and_translate(text):
    """
    Detect language and translate to English if needed.

    Thresholds:
    - MIN_LENGTH: 30 chars (skip detection on very short text, assume English)
    - MIN_CONFIDENCE: 0.85 (only translate if confidence >= 85%)

    Returns: (translated_text, detected_language)
    Falls back to original text on translation failure or low confidence.
    """
    MIN_LENGTH = 30
    MIN_CONFIDENCE = 0.85

    # Skip detection for very short texts (too unreliable with emojis, etc.)
    if len(text.strip()) < MIN_LENGTH:
        return text, "en"

    try:
        # Get confidence scores for all detected languages
        results = detect_langs(text)
        top_detection = results[0]

        # If top detection is English, no translation needed
        if top_detection.lang == "en":
            return text, "en"

        # If confidence is below threshold, assume English (fall back)
        if top_detection.prob < MIN_CONFIDENCE:
            return text, "en"

        # Translate non-English text with high confidence
        translator = GoogleTranslator(source_language=top_detection.lang, target_language="en")
        translated = translator.translate(text)
        return translated, top_detection.lang

    except LangDetectException:
        # If detection fails, assume English
        return text, "en"
    except Exception as e:
        # On translation failure, fall back to original text
        return text, "en"

# ---- MAIN PIPELINE ----
total = len(posts)
print(f"Processing {total} records...")

enriched = []

for i, post in enumerate(posts, 1):
    original_text = post["text"]

    # Language detection and translation
    text, detected_language = detect_and_translate(original_text)

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

    enriched.append({
        **post,
        "text": text,  # Store translated text for analysis
        "original_text": original_text,  # Keep original
        "detected_language": detected_language,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "emotional_affinity_score": emotional_affinity_score,
        "behavioral_pathway": behavioral_pathway,
        "priority_signal": priority_signal,
        "confidence_score": confidence_score,
    })

    print(f"  [{i}/{total}] {str(post['post_id'])[:8]}... {sentiment} | {behavioral_pathway} | {priority_signal} | lang: {detected_language}")

# ---- SAVE OUTPUT ----
with open(OUTPUT_PATH, "w") as f:
    json.dump(enriched, f, indent=2)

print(f"\nDone. {total} records saved to {OUTPUT_PATH}")

# ---- QUICK SUMMARY ----
sentiments = [r["sentiment"] for r in enriched]
print(f"\nSentiment breakdown:")
for label in ("positive", "neutral", "negative"):
    print(f"  {label}: {sentiments.count(label)}")