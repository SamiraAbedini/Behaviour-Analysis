import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/behavior/output_file.csv')

# Preprocess the data
df = df.drop(columns=['ID'])  # Drop ID column as it's not useful
df_cleaned = df.dropna(axis=1)  # Drop columns with NaN values
print("Remaining columns:", df_cleaned.columns)

# Convert 'Group' column: Merge groups if required
df_cleaned['Group'] = df_cleaned['Group'].replace(3, 2)

# Split features and target variable
X = df_cleaned.drop('Group', axis=1)
y = df_cleaned['Group']

# One-hot encode the target variable for neural networks
y_one_hot = to_categorical(y - 1)  # Classes start from 0

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to evaluate and display model performance
def evaluate_model(y_true, y_pred, y_pred_prob=None, classes=2):
    accuracy = accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    conf_matrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    class_report = classification_report(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    print(f'Accuracy: {accuracy:.2f}')
    print('Confusion Matrix:\n', conf_matrix)
    print('Classification Report:\n', class_report)

    if y_pred_prob is not None:
        plt.figure(figsize=(8, 6))
        for i in range(classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
            roc_auc = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
            plt.plot(fpr, tpr, label=f'Class {i + 1} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curve')
        plt.legend()
        plt.show()

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.argmax(axis=1))
rf_preds = rf_model.predict_proba(X_test)
evaluate_model(y_test, np.eye(rf_preds.shape[1])[rf_preds.argmax(axis=1)], rf_preds)

# Neural Network Model
nn_model = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')
])

nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = nn_model.fit(
    X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[early_stopping], verbose=1
)

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate Neural Network
nn_preds = nn_model.predict(X_test)
evaluate_model(y_test, nn_preds, nn_preds)
