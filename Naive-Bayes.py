from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

file_path = 'Path to file'# Set the path to CSV
df = pd.read_csv(file_path)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_vec = vectorizer.fit_transform(X_train)
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
all_X_vec = vectorizer.transform(X)
all_predictions = nb_model.predict(all_X_vec)
for i, (review, true_label, predicted_label) in enumerate(zip(X, y, all_predictions)):
    accuracy = 1 if true_label == predicted_label else 0
    sentiment_label = 'позитивное' if predicted_label == 1 else 'негативное'
    print(f"Review {i + 1}: {review[:60]}... | Predicted Sentiment: {sentiment_label} | Accuracy: {accuracy}")
overall_accuracy = accuracy_score(y, all_predictions)
print(f'\nOverall Accuracy for all reviews: {overall_accuracy:.2f}')