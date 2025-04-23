import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def load_mnist(kind='train'):
    """Load MNIST data from uncompressed raw files"""
    labels_path = f'{kind}-labels.idx1-ubyte'
    images_path = f'{kind}-images.idx3-ubyte'
    
    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    
    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                             offset=16).reshape(len(labels), 28, 28)
    
    return images, labels

def load_and_prepare_data():
    """Load and prepare MNIST dataset"""
    X_train, y_train = load_mnist(kind='train')
    X_test, y_test = load_mnist(kind='t10k')
    
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    X_train = X_train[..., tf.newaxis]
    X_test = X_test[..., tf.newaxis]
    
    return X_train, y_train, X_test, y_test

def create_model():
    """Create CNN model"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model(model, train_images, train_labels, epochs=5):
    """Train model and save plots"""
    history = model.fit(train_images, train_labels, epochs=epochs, 
                       validation_split=0.2, batch_size=64, verbose=1)
    
    # Save training history plot without showing
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.9, 1.0])
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()  # Close figure
    
    return model

def evaluate_model(model, test_images, test_labels):
    """Evaluate model and save prediction samples"""
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")
    
    predictions = model.predict(test_images, verbose=0)
    
    # Save prediction samples without showing
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions[i])
        true_label = test_labels[i]
        color = 'green' if predicted_label == true_label else 'red'
        plt.xlabel(f"Pred: {predicted_label}\nTrue: {true_label}", color=color)
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()  # Close figure 

def save_model(model):
    """Save model in recommended .keras format"""
    model.save('mnist_model.keras')  # Using new recommended format
    print("Model saved as mnist_model.keras")

def main():
    print("Loading MNIST dataset...")
    train_images, train_labels, test_images, test_labels = load_and_prepare_data()
    
    print("\nCreating model...")
    model = create_model()
    model.summary()
    
    print("\nTraining model...")
    model = train_model(model, train_images, train_labels, epochs=5)
    
    print("\nEvaluating model...")
    evaluate_model(model, test_images, test_labels)
    
    save_model(model)
    print("\nDone! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    # Set matplotlib to non-interactive backend
    plt.switch_backend('Agg')
    main()