import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

data = {
    'text': [
        "Win money now!", "Hi, how are you?", "Free investment opportunity", 
        "Meeting at 2pm", "Click here for a prize", "Can we reschedule?",
        "URGENT: Your account is locked", "Happy Birthday!"
    ],
    'label': ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"]
}
df = pd.DataFrame(data)

print("Training model...")
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df['text'], df['label'])

joblib.dump(model, "spam_model.pkl")
print("Success! Model saved as 'spam_model.pkl'")