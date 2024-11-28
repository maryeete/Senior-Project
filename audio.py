import sys
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import librosa

def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    
    # Add more relevant features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    
    features = np.hstack([
        np.mean(mfccs, axis=1),
        np.std(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        np.mean(mel, axis=1),
        np.std(mel, axis=1),
        np.mean(contrast, axis=1),
        np.std(contrast, axis=1),
        np.mean(tonnetz, axis=1),
        np.std(tonnetz, axis=1),
        np.mean(zcr),
        np.std(zcr),
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth)
    ])
    
    return features

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try different models
    models = {
        'SVM': SVC(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'MLP': MLPClassifier(random_state=42)
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        mean_cv_score = np.mean(cv_scores)
        print(f"{name} Cross-validation score: {mean_cv_score:.3f}")
        
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_model = model
    
    # Fine-tune hyperparameters for the best model
    if isinstance(best_model, SVC):
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
    elif isinstance(best_model, RandomForestClassifier):
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    elif isinstance(best_model, MLPClassifier):
        param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]}
    
    grid_search = GridSearchCV(best_model, param_grid, cv=5)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best model: {type(best_model).__name__}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    
    return scaler, best_model

def predict_emotion(audio_file, scaler, model):
    features = extract_features(audio_file)
    features_scaled = scaler.transform(features.reshape(1, -1))
    emotion = model.predict(features_scaled)[0]
    return emotion

def sort_audio_files(input_folder, output_folder, scaler, model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for emotion in ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']:
        emotion_folder = os.path.join(output_folder, emotion)
        if not os.path.exists(emotion_folder):
            os.makedirs(emotion_folder)
    
    for audio_file in os.listdir(input_folder):
        if audio_file.endswith('.wav'):
            file_path = os.path.join(input_folder, audio_file)
            try:
                predicted_emotion = predict_emotion(file_path, scaler, model)
                destination = os.path.join(output_folder, predicted_emotion, audio_file)
                shutil.copy(file_path, destination)
                print(f"Sorted {audio_file} as {predicted_emotion}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

def main():
    archive_path = os.path.expanduser('~/Downloads/sentiment_analysis/archive')
    sorted_dataset_path = os.path.expanduser('~/sorted_emotion_audio_dataset')
    
    if not os.path.exists(archive_path):
        print(f"Error: The archive path '{archive_path}' does not exist.")
        sys.exit(1)
    
    # Get all actor folders
    actor_folders = [f for f in os.listdir(archive_path) if f.startswith('Actor_')]
    
    if not actor_folders:
        print(f"Error: No Actor folders found in {archive_path}")
        sys.exit(1)
    
    # Interactive selection process
    print("Available actor folders:")
    for i, folder in enumerate(actor_folders, 1):
        print(f"{i}. {folder}")
    print(f"{len(actor_folders) + 1}. Process all actors")
    
    while True:
        try:
            choice = int(input("Enter the number of the actor folder you want to process (or select 'Process all actors'): "))
            if 1 <= choice <= len(actor_folders) + 1:
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    if choice == len(actor_folders) + 1:
        selected_actors = actor_folders
    else:
        selected_actors = [actor_folders[choice - 1]]
    
    X = []
    y = []
    
    # Process selected actors
    for actor_folder in selected_actors:
        actor_path = os.path.join(archive_path, actor_folder)
        audio_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
        
        for audio_file in audio_files:  # Process all files from each actor for training
            file_path = os.path.join(actor_path, audio_file)
            try:
                features = extract_features(file_path)
                X.append(features)
                emotion = audio_file.split('-')[2]
                emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', 
                               '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
                y.append(emotion_map.get(emotion, 'unknown'))
                print(f"Processed for training: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    if not X:
        print("Error: No audio files were successfully processed for training.")
        sys.exit(1)

    X = np.array(X)
    y = np.array(y)
    
    scaler, model = train_model(X, y)
    
    # Sort files from selected actors
    for actor_folder in selected_actors:
        actor_path = os.path.join(archive_path, actor_folder)
        actor_sorted_path = os.path.join(sorted_dataset_path, actor_folder)
        sort_audio_files(actor_path, actor_sorted_path, scaler, model)
    
    print("Audio files have been sorted. You can find them in:", sorted_dataset_path)

if __name__ == "__main__":
    main()