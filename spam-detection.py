import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
import warnings

# Load the dataset
data = pd.read_csv(r"C:\Users\Agniv\Desktop\Internships\devtern internship\machine learning project\Spam Detection project\dataset\spam_ham_dataset.csv", encoding='latin-1')

# Handling missing values
data.fillna(method='ffill', inplace=True)

# Select columns for analysis
data = data[['label', 'message']]

# Encode labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Convert 'message' column to strings
X_train = X_train.astype(str)
X_test = X_test.astype(str)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Build the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train.toarray(), y_train, epochs=15, batch_size=64, validation_data=(X_test.toarray(), y_test))

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

# Evaluate the model
y_pred = model.predict_classes(X_test.toarray())
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=1)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')










