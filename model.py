import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load preprocessed data
train_df = pd.read_csv('Train_Preprocess.csv')
test_df = pd.read_csv('Test_Preprocess.csv')

# 2. Create synthetic time-series structure for CNN
def create_sequences(data, seq_length=10):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length, :-1].values
        label = data.iloc[i+seq_length, -1]
        sequences.append(seq)
        targets.append(label)
    return np.array(sequences), np.array(targets)

SEQ_LENGTH = 10  # Length of each time sequence
FEATURES = train_df.shape[1] - 1  # Number of features

# Create time sequences for training data
X_train, y_train = create_sequences(train_df, SEQ_LENGTH)

# Create time sequences for test data
X_test, y_test = create_sequences(test_df, SEQ_LENGTH)

# 3. Reshape data for CNN input (add channel dimension)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# 4. Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# 5. Build CNN model
model = Sequential([
    # First convolution block
    Conv1D(64, kernel_size=3, activation='relu', 
           input_shape=(SEQ_LENGTH, FEATURES, 1), padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    
    # Second convolution block
    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),
    
    # Third convolution block
    Conv1D(256, kernel_size=2, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.5),
    
    # Fully connected layers
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(1, activation='sigmoid')
])

# 6. Compile the model
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.AUC(name='auc')])

# 7. Define callbacks
callbacks = [
    EarlyStopping(monitor='val_auc', patience=10, mode='max', 
                  restore_best_weights=True),
    ModelCheckpoint('best_cnn_model.h5', monitor='val_auc', 
                   save_best_only=True, mode='max')
]

# 8. Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    class_weight={0: 1, 1: 2.2}  # Higher weight for positive class
)

# 9. Evaluate the model
def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model on test data:")
    results = model.evaluate(X_test, y_test)
    print(f"Final Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]:.4f}")
    print(f"Precision: {results[2]:.4f}")
    print(f"Recall: {results[3]:.4f}")
    print(f"AUC: {results[4]:.4f}")
    
    # Predictions and classification report
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Backorder', 'Backorder'],
                yticklabels=['No Backorder', 'Backorder'])
    plt.xlabel('Model Prediction')
    plt.ylabel('Actual Value')
    plt.title('CNN Model - Confusion Matrix')
    plt.show()

evaluate_model(model, X_test, y_test)

# 10. Plot training history
def plot_training_history(history):
    plt.figure(figsize=(14,5))
    
    # Accuracy plot
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Progress')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epoch')
    plt.legend()
    
    # AUC plot
    plt.subplot(1,2,2)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Model AUC Progress')
    plt.ylabel('AUC')
    plt.xlabel('Training Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# 11. Save final model
model.save('backorder_cnn_model.h5')
print("CNN model saved successfully!")
