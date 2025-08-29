import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification, create_optimizer
import os

# 1. Load dataset
file_path = "customer_sentiment.csv"
df = pd.read_csv(file_path, header=None, names=['ID', 'Company', 'Label', 'Comment'])
print(df.head())

# 2. Preprocess dataset
df['Comment'] = df['Comment'].fillna('No Comment')
# Map labels to numeric classes
label_mapping = {"Positive": 0, "Negative": 1, "Neutral": 2, "Irrelevant": 3}
df['Label'] = df['Label'].map(label_mapping)
# Drop rows with unmapped labels (if any)
df = df.dropna(subset=['Label'])

# Features and labels
X = df['Comment'].tolist()
y = df['Label'].astype(int).tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128, return_tensors="tf")
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128, return_tensors="tf")

# Convert to TF tensors
train_encodings_tf = {k: tf.convert_to_tensor(v.numpy()) for k, v in train_encodings.items()}
test_encodings_tf = {k: tf.convert_to_tensor(v.numpy()) for k, v in test_encodings.items()}

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_encodings_tf, y_train)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((test_encodings_tf, y_test)).batch(16)

# 4. Define and Compile Model
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

batch_size = 16
epochs = 3
steps_per_epoch = len(train_dataset)
num_train_steps = steps_per_epoch * epochs

optimizer, schedule = create_optimizer(init_lr=2e-5, num_train_steps=num_train_steps, num_warmup_steps=0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy("accuracy")]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print("Model compiled successfully")

# 5. Train & Evaluate
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)

results = model.evaluate(test_dataset)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# 6. Predictions & Reports
y_pred_logits = model.predict(test_dataset).logits
y_pred = tf.argmax(y_pred_logits, axis=1).numpy()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Save Model
save_dir = "sentiment_model"
os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Model and tokenizer saved at: {os.path.abspath(save_dir)}")
