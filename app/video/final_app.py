import sys
import os
import glob
import time
import platform

from flask import Blueprint
try:
    import cv2
    import numpy as np
    from fer import FER
    import librosa
    import sounddevice as sd
    import threading
    import queue
    from collections import deque
    from PIL import Image
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt
    import joblib
    import warnings
    warnings.filterwarnings('ignore')
except ImportError:
    print("❌ Required packages are not installed.")
    print("Please run setup.py first to install dependencies.")
    sys.exit(1)

auth = Blueprint('seb1', __name__)

class EmotionAnalyzer:
    def __init__(self):
        # Set platform-specific configurations
        self.os_type = platform.system()  # Will return 'Darwin' for Mac, 'Windows' for Windows
        self.emotion_detector = FER(mtcnn=True)
        self.audio_queue = queue.Queue()
        self.running = True
        self.sample_rate = 22050
        self.audio_window = 2
        self.visual_emotions = deque(maxlen=5)
        self.audio_emotions = deque(maxlen=5)
        self.cap = None
        self.frame_skip = 2
        self.frame_count = 0
        self.last_result = None
        self.frame_length = 2048
        self.hop_length = 512
        self.scaler = StandardScaler()
        self.model_path = "audio_emotion_model.joblib"
        self.dataset_path = "archive"
        self.selected_audio_device = None
        self.selected_video_device = None
        try:
            self.audio_classifier = joblib.load(self.model_path)
            self.scaler = joblib.load("audio_scaler.joblib")
            print("✅ Loaded pre-trained audio emotion model")
        except:
            print("⚙️ Training new audio emotion model...")
            self.train_audio_model()

    def select_audio_device(self):
        """Let user select audio input device"""
        print("\n🎤 Available Audio Devices:")
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # Only show input devices
                input_devices.append((i, device))
                print(f"{len(input_devices)-1}: {device['name']}")
        
        while True:
            choice = input("\nSelect audio device number (or press Enter for default): ").strip()
            if choice == "":
                return None
            try:
                idx = int(choice)
                if 0 <= idx < len(input_devices):
                    return input_devices[idx][0]
                print("❌ Invalid device number")
            except ValueError:
                print("❌ Please enter a valid number")

    def select_video_device(self):
        """Let user select video input device"""
        print("\n📹 Available Video Devices:")
        available_cameras = []
        
        # Test the first 5 camera indices
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                    # Show preview
                    small_frame = cv2.resize(frame, (320, 240))
                    cv2.imshow(f'Camera {i}', small_frame)
                    cv2.waitKey(1000)
                cap.release()
                cv2.destroyAllWindows()
        
        if not available_cameras:
            print("❌ No cameras found")
            return 0
            
        print("\nDetected cameras:")
        for i in available_cameras:
            print(f"{i}: Camera {i}")
            
        while True:
            choice = input("\nSelect camera number (or press Enter for default): ").strip()
            if choice == "":
                return 0
            try:
                idx = int(choice)
                if idx in available_cameras:
                    return idx
                print("❌ Invalid camera number")
            except ValueError:
                print("❌ Please enter a valid number")

    def initialize_camera(self):
        """Platform-independent camera initialization"""
        print("🎥 Initializing camera...")
        
        camera_index = self.select_video_device()
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print("❌ Could not access webcam")
            return False
            
        # Optimize camera settings for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Additional performance settings
        if self.os_type != 'Darwin':  # Non-macOS
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
            self.cap.set(cv2.CAP_PROP_SETTINGS, 0)   # Disable auto settings
        
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"📊 Camera FPS: {actual_fps}")
        
        return True

    def analyze_image(self, image_path):
        """Analyze emotions in a single image"""
        try:
            print(f"🖼️ Analyzing image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")

            # Detect emotions
            result = self.emotion_detector.detect_emotions(image)
            
            if not result:
                print("❌ No faces detected in the image")
                return

            # Process each face in the image
            for i, face in enumerate(result):
                emotions = face['emotions']
                x, y, w, h = face['box']
                
                # Draw rectangle around face
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Get dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                confidence = emotions[dominant_emotion]
                
                # Display emotion text
                text = f"Face {i+1}: {dominant_emotion} ({confidence:.2f})"
                cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, (0, 255, 0), 2)
                
                # Print results
                print(f"\n📊 Results for Face {i+1}:")
                for emotion, score in emotions.items():
                    print(f"{emotion}: {score:.2f}")

            # Show the annotated image
            cv2.imshow('Image Analysis (Press any key to close)', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"❌ Error analyzing image: {e}")

    def extract_audio_features(self, audio_data):
        """Extract comprehensive audio features"""
        try:
            # Convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            features = {}
            
            # 1. MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, 
                                       n_mfcc=20)  # Increased from 13 to 20
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # 2. Spectral Features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # 3. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 4. Root Mean Square Energy
            rms = librosa.feature.rms(y=audio_data)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # 5. Speech Rate (using zero crossings as proxy)
            features['speech_rate'] = np.sum(np.diff(np.signbit(audio_data)))
            
            # 6. Pitch and Intensity
            f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, 
                                                       fmin=librosa.note_to_hz('C2'), 
                                                       fmax=librosa.note_to_hz('C7'))
            features['pitch_mean'] = np.mean(f0[~np.isnan(f0)]) if any(~np.isnan(f0)) else 0
            features['pitch_std'] = np.std(f0[~np.isnan(f0)]) if any(~np.isnan(f0)) else 0
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return None

    def classify_audio_emotion(self, features):
        """Enhanced audio emotion classification"""
        if features is None:
            return "neutral"
            
        try:
            # Convert features to vector
            feature_vector = []
            for feature_group in features.values():
                if isinstance(feature_group, np.ndarray):
                    feature_vector.extend(feature_group.flatten())
                else:
                    feature_vector.append(feature_group)
            
            # Scale features
            scaled_features = self.scaler.transform([feature_vector])
            
            # Get prediction probabilities
            proba = self.audio_classifier.predict_proba(scaled_features)[0]
            emotions = self.audio_classifier.classes_
            emotion_probs = list(zip(emotions, proba))
            
            # Sort by probability
            emotion_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Get top emotion and its probability
            top_emotion, top_prob = emotion_probs[0]
            
            # Higher confidence threshold
            if top_prob > 0.4:  # Adjusted threshold
                # Additional checks for specific emotions
                if top_emotion == "neutral" and top_prob < 0.6:
                    # Check second highest emotion
                    if len(emotion_probs) > 1:
                        second_emotion, second_prob = emotion_probs[1]
                        if second_prob > 0.3:  # If second emotion is significant
                            return second_emotion
                
                print(f"Audio Emotion: {top_emotion} ({top_prob:.2f})")
                return top_emotion
            else:
                return "neutral"
            
        except Exception as e:
            print(f"Error in classification: {e}")
            return "neutral"

    def run_combined_analysis(self):
        """Run simultaneous video and audio analysis"""
        if not self.initialize_camera():
            return

        print("\n🚀 Starting Combined Analysis")
        print("Press 'Q' to quit the analysis")
        
        # Start audio thread
        audio_thread = threading.Thread(target=self.audio_analysis_thread, daemon=True)
        audio_thread.start()

        # Initialize variables
        self.frame_buffer = deque(maxlen=2)
        self.last_emotion = None
        self.emotion_update_counter = 0
        self.emotion_update_frequency = 30  # Process emotions every 30 frames
        last_time = time.time()
        fps = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                current_time = time.time()
                fps = 1 / (current_time - last_time)
                last_time = current_time

                frame = cv2.flip(frame, 1)
                
                # Process emotions less frequently
                if self.emotion_update_counter % self.emotion_update_frequency == 0:
                    # Process in a separate thread to prevent blocking
                    emotion_thread = threading.Thread(
                        target=self.process_emotion_async,
                        args=(frame.copy(),),
                        daemon=True
                    )
                    emotion_thread.start()

                # Use last known emotion result
                if self.last_emotion:
                    self.process_video_frame(frame, self.last_emotion)

                # Display audio emotion
                if self.audio_emotions:
                    current_audio_emotion = self.audio_emotions[-1]
                    # Use numpy operations instead of cv2.rectangle for speed
                    overlay = frame[10:100, 10:400].copy()
                    cv2.addWeighted(
                        np.full_like(overlay, 0),  # black rectangle
                        0.7,  # alpha
                        overlay,
                        0.3,  # beta
                        0,  # gamma
                        overlay
                    )
                    frame[10:100, 10:400] = overlay
                    
                    cv2.putText(frame, f"Audio: {current_audio_emotion}",
                              (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                              1.0, (255, 255, 255), 2)
                    cv2.putText(frame, f"FPS: {fps:.1f}",
                              (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                              1.0, (255, 255, 255), 2)

                cv2.imshow('Combined Analysis', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                self.emotion_update_counter += 1

        finally:
            self.cleanup()

    def process_emotion_async(self, frame):
        """Process emotions asynchronously"""
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Reduce size more
        result = self.emotion_detector.detect_emotions(small_frame)
        if result:
            # Adjust coordinates back to original frame size
            for face in result:
                face['box'] = [x * 4 for x in face['box']]
            self.last_emotion = result

    def run_video_only(self):
        """Run video-only analysis"""
        if not self.initialize_camera():
            return

        print("\n🎥 Starting Video-Only Analysis")
        print("Press 'Q' to quit")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                
                # Only process every nth frame
                if self.frame_count % self.frame_skip == 0:
                    result = self.emotion_detector.detect_emotions(frame)
                    if result:
                        self.last_result = result
                self.frame_count += 1

                # Use cached results for drawing
                if self.last_result:
                    self.process_video_frame(frame, self.last_result)

                cv2.imshow('Video Analysis (Press Q to quit)', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cleanup()

    def process_video_frame(self, frame, result):
        """Optimized frame processing"""
        for face in result:
            emotions = face['emotions']
            x, y, w, h = face['box']
            
            # Calculate dominant emotion only once
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, confidence = dominant_emotion
            
            # Draw rectangle and text efficiently
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add background rectangle for better text visibility
            text = f"{emotion_name}: {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(frame, 
                         (x, max(y-10-text_size[1], 0)), 
                         (x + text_size[0], max(y-10, 0)), 
                         (0, 255, 0), 
                         -1)
            cv2.putText(frame, text,
                       (x, max(y-10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            
            self.visual_emotions.append(emotion_name)

    def cleanup(self):
        """Platform-independent cleanup"""
        print("\n🧹 Cleaning up...")
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Additional cleanup for macOS
        if self.os_type == 'Darwin':
            for i in range(5):  # Sometimes needed on macOS
                cv2.waitKey(1)

    def audio_analysis_thread(self):
        """Platform-independent audio analysis"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            self.audio_queue.put(indata.copy())

        try:
            # Platform-specific audio settings
            if self.os_type == 'Darwin':
                buffer_size = int(self.sample_rate * 0.25)  # Smaller buffer for macOS
                silence_threshold = 0.01
            else:
                buffer_size = int(self.sample_rate * 0.5)  # Windows buffer
                silence_threshold = 0.005  # Lower threshold for Windows

            # Initialize emotion window for smoothing
            emotion_window = deque(maxlen=5)
            last_active_time = time.time()
            
            # Get user selected audio device
            device_index = self.select_audio_device()

            with sd.InputStream(callback=audio_callback,
                              channels=1,
                              samplerate=self.sample_rate,
                              blocksize=buffer_size,
                              dtype=np.float32,
                              device=device_index):
                
                print("🎤 Audio analysis started")
                while self.running:
                    try:
                        if not self.audio_queue.empty():
                            audio_data = self.audio_queue.get()
                            
                            # Check if audio is silent
                            audio_level = np.abs(audio_data).mean()
                            
                            if audio_level < silence_threshold:
                                emotion_window.clear()  # Clear emotion history
                                self.audio_emotions.append("silent")
                                continue
                            
                            # Process audio only if not silent
                            features = self.extract_audio_features(audio_data.flatten())
                            
                            if features:
                                emotion = self.classify_audio_emotion(features)
                                
                                # Add to emotion window only if confident
                                if emotion != "neutral":
                                    emotion_window.append(emotion)
                                
                                # Use most common emotion in window with minimum count
                                if emotion_window:
                                    from collections import Counter
                                    emotion_counts = Counter(emotion_window)
                                    most_common = emotion_counts.most_common(1)[0]
                                    
                                    # Only use emotion if it appears multiple times
                                    if most_common[1] >= 2:  # Minimum 2 occurrences
                                        smoothed_emotion = most_common[0]
                                    else:
                                        smoothed_emotion = "neutral"
                                    
                                    self.audio_emotions.append(smoothed_emotion)
                                else:
                                    self.audio_emotions.append("neutral")
                    except Exception as e:
                        print(f"Error processing audio: {e}")
                        continue

        except Exception as e:
            print(f"Audio thread error: {e}")
            if self.os_type == 'Darwin':
                print("\n⚠️ On macOS, make sure to grant microphone permissions to your terminal/IDE")
            else:
                print("\n⚠️ Make sure your microphone is properly connected and not in use by another application")
            print("Try running the program from a different terminal or IDE")

    def extract_audio_features_enhanced(self, audio_data):
        """Enhanced audio feature extraction with noise reduction"""
        try:
            # Noise reduction
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Apply noise gate
            noise_gate = 0.01
            audio_data[np.abs(audio_data) < noise_gate] = 0
            
            # Normalize audio
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
            
            # Skip if audio is too quiet
            if np.max(np.abs(audio_data)) < 0.1:
                return None
            
            features = {}
            
            # Enhanced feature extraction
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, 
                                       n_mfcc=13, hop_length=self.hop_length)
            
            # Only use stable parts of the features
            features['mfcc_mean'] = np.mean(mfccs[:, 5:-5], axis=1)  # Skip edges
            features['mfcc_std'] = np.std(mfccs[:, 5:-5], axis=1)
            
            # Spectral features with stability checks
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate, hop_length=self.hop_length)[0]
            
            if len(spectral_centroids) > 10:  # Ensure enough frames
                features['spectral_centroid_mean'] = np.mean(spectral_centroids[5:-5])
                features['spectral_centroid_std'] = np.std(spectral_centroids[5:-5])
            else:
                return None
            
            # Energy features with threshold
            rms = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)[0]
            if np.mean(rms) > 0.01:  # Energy threshold
                features['rms_mean'] = np.mean(rms)
                features['rms_std'] = np.std(rms)
            else:
                return None
            
            return features
            
        except Exception as e:
            return None

    def classify_audio_emotion_enhanced(self, features):
        """Enhanced emotion classification with confidence checks"""
        if features is None:
            return "neutral"
            
        try:
            # Convert features to vector
            feature_vector = []
            for feature_group in features.values():
                if isinstance(feature_group, np.ndarray):
                    feature_vector.extend(feature_group.flatten())
                else:
                    feature_vector.append(feature_group)
            
            # Scale features
            scaled_features = self.scaler.transform([feature_vector])
            
            # Get prediction probabilities
            proba = self.audio_classifier.predict_proba(scaled_features)[0]
            max_proba = np.max(proba)
            
            # Higher confidence threshold
            if max_proba > 0.5:  # Increased from 0.4
                emotion = self.audio_classifier.predict(scaled_features)[0]
                
                # Additional checks for specific emotions
                if emotion == "fearful" and max_proba < 0.7:  # Higher threshold for fearful
                    return "neutral"
                    
                return emotion
            else:
                return "neutral"
            
        except Exception as e:
            return "neutral"

    def validate_dataset(self):
        """Validate and analyze the dataset structure"""
        print("\n📊 Dataset Analysis:")
        emotion_counts = {}
        valid_files = 0
        invalid_files = 0
        
        for emotion_folder in glob.glob(f"{self.dataset_path}/*"):
            emotion = os.path.basename(emotion_folder)
            if os.path.isdir(emotion_folder):
                audio_files = glob.glob(f"{emotion_folder}/*.wav")
                count = len(audio_files)
                emotion_counts[emotion] = count
                valid_files += count
                
                # Validate first file in each folder
                if audio_files:
                    try:
                        audio_data, sr = librosa.load(audio_files[0], sr=self.sample_rate)
                        print(f"✅ {emotion}: {count} files (Sample validated)")
                    except Exception as e:
                        print(f"⚠️ {emotion}: {count} files (Sample validation failed: {e})")
                        invalid_files += count
                else:
                    print(f"⚠️ {emotion}: No WAV files found")

        print(f"\nTotal valid files: {valid_files - invalid_files}")
        if invalid_files > 0:
            print(f"Total invalid files: {invalid_files}")
            
        return emotion_counts

    def train_audio_model(self):
        """Train model using RAVDESS dataset with enhanced accuracy"""
        features = []
        labels = []
        print("\n🎯 Processing audio files...")
        
        # RAVDESS emotion mapping
        emotion_map = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
        processed_files = 0
        
        for actor_folder in glob.glob(f"{self.dataset_path}/Actor_*"):
            for audio_file in glob.glob(f"{actor_folder}/*.wav"):
                try:
                    # Parse emotion from filename
                    filename = os.path.basename(audio_file)
                    emotion_code = filename.split('-')[2]
                    emotion = emotion_map.get(emotion_code)
                    
                    if not emotion:
                        continue
                    
                    # Load and preprocess audio
                    audio_data, _ = librosa.load(audio_file, sr=self.sample_rate, duration=3.0)
                    
                    # Extract features
                    audio_features = self.extract_audio_features(audio_data)
                    
                    if audio_features:
                        feature_vector = []
                        for feature_group in audio_features.values():
                            if isinstance(feature_group, np.ndarray):
                                feature_vector.extend(feature_group.flatten())
                            else:
                                feature_vector.append(feature_group)
                        
                        features.append(feature_vector)
                        labels.append(emotion)
                        processed_files += 1
                        
                        print(f"\rProcessing: {processed_files} files", end="")
                    
                except Exception as e:
                    continue

        if not features:
            print("❌ No valid features extracted")
            return

        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model with better parameters
        print("\n🚀 Training model...")
        self.audio_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        
        self.audio_classifier.fit(X_train_scaled, y_train)

        # Save model and scaler
        joblib.dump(self.audio_classifier, self.model_path)
        joblib.dump(self.scaler, "audio_scaler.joblib")

    def validate_features(self, features):
        """Validate extracted features"""
        try:
            # Check for NaN or infinite values
            for feature_group in features.values():
                if isinstance(feature_group, np.ndarray):
                    if np.isnan(feature_group).any() or np.isinf(feature_group).any():
                        return False
                elif np.isnan(feature_group) or np.isinf(feature_group):
                    return False
            return True
        except:
            return False

    def analyze_audio_file(self, audio_path):
        """Analyze emotions in an audio file"""
        try:
            print(f"🎵 Analyzing audio file: {audio_path}")
            
            # Load audio file
            audio_data, _ = librosa.load(audio_path, sr=self.sample_rate)
            
            # Split audio into chunks for analysis
            chunk_length = int(self.sample_rate * 3)  # 3 second chunks
            chunks = [audio_data[i:i + chunk_length] 
                     for i in range(0, len(audio_data), chunk_length)]
            
            emotions = []
            print("\n📊 Analyzing emotions in audio...")
            
            for i, chunk in enumerate(chunks):
                if len(chunk) < self.sample_rate:  # Skip chunks shorter than 1 second
                    continue
                    
                # Extract features from chunk
                features = self.extract_audio_features(chunk)
                if features:
                    # Classify emotion
                    emotion = self.classify_audio_emotion(features)
                    emotions.append(emotion)
                    
                    # Show progress
                    progress = ((i + 1) / len(chunks)) * 100
                    print(f"\rProgress: {progress:.1f}%", end="")
            
            print("\n\n🎯 Results:")
            if emotions:
                # Count occurrences of each emotion
                from collections import Counter
                emotion_counts = Counter(emotions)
                
                # Calculate percentages
                total = sum(emotion_counts.values())
                for emotion, count in emotion_counts.most_common():
                    percentage = (count / total) * 100
                    print(f"{emotion}: {percentage:.1f}%")
                
                # Plot results
                plt.figure(figsize=(10, 6))
                emotions_list = list(emotion_counts.keys())
                counts_list = list(emotion_counts.values())
                
                plt.bar(emotions_list, counts_list)
                plt.title('Emotion Distribution in Audio')
                plt.xlabel('Emotion')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save plot
                plot_path = "audio_analysis_result.png"
                plt.savefig(plot_path)
                print(f"\n📈 Plot saved as: {plot_path}")
                
                # Show plot
                plt.show()
                
            else:
                print("❌ No emotions detected in the audio file")
            
        except Exception as e:
            print(f"❌ Error analyzing audio file: {e}")
            import traceback
            traceback.print_exc()

def main():
    analyzer = EmotionAnalyzer()
    
    while True:
        print("\n🤖 Emotion Analyzer Menu")
        print("1. Analyze Image")
        print("2. Analyze Audio File")
        print("3. Real-time Video Analysis")
        print("4. Combined Audio-Video Analysis")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            image_path = input("Enter the path to your image file: ")
            if os.path.exists(image_path):
                analyzer.analyze_image(image_path)
            else:
                print("❌ Invalid image path")
                
        elif choice == '2':
            audio_path = input("Enter the path to your audio file: ")
            if os.path.exists(audio_path):
                analyzer.analyze_audio_file(audio_path)
            else:
                print("❌ Invalid audio path")
                
        elif choice == '3':
            analyzer.run_video_only()
            
        elif choice == '4':
            analyzer.run_combined_analysis()
            
        elif choice == '5':
            print("\n👋 Thank you for using Emotion Analyzer!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Please make sure your camera and microphone are properly connected.")
        input("\nPress Enter to exit...")