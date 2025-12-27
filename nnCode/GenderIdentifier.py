import os
import pickle
import warnings
import numpy as np
from hmmlearn import hmm
from nnCode.FeaturesExtractor import FeaturesExtractor
import tensorflow as tf # Replaces generic keras import for better compatibility
from tensorflow import keras

warnings.filterwarnings("ignore")

class GenderIdentifier:

    def __init__(self, females_files_path, males_files_path, females_model_path, males_model_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.error                 = 0
        self.total_sample          = 0
        self.features_extractor    = FeaturesExtractor()
        
        # 1. LOAD MODELS
        print("Loading GMM models...")
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm   = pickle.load(open(males_model_path, 'rb'))
        
        # 2. PREPARE DATA FOR NEURAL NETWORK
        # The Neural Network cannot read the GMM object directly. 
        # We must extract the 'means_' (the 16 vectors representing the voice).
        
        # Shape of means_: (16, 39) -> 16 components, 39 features each
        female_vectors = self.females_gmm.means_
        male_vectors   = self.males_gmm.means_
        
        # Stack them to create training data
        self.X_train = np.vstack((female_vectors, male_vectors))
        
        # Create labels: 0 for Female, 1 for Male
        # We have 16 female vectors and 16 male vectors
        self.y_train = np.hstack((np.zeros(len(female_vectors)), np.ones(len(male_vectors))))
        
        print(f"NN Training Data Shape: {self.X_train.shape}") # Should be (32, 39)
        
        # 3. DEFINE & TRAIN KERAS MODEL
        # We train the NN to classify these specific vectors
        print("Training Neural Network...")
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(39, input_dim=39, activation='relu'))
        self.model.add(keras.layers.Dense(13, activation='relu'))
        self.model.add(keras.layers.Dense(2, activation='softmax')) # Softmax is better for categorical
        
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy', # Matches integer labels (0, 1)
                           metrics=['accuracy'])
                           
        self.model.fit(self.X_train, self.y_train, epochs=50, verbose=0)
        print("Neural Network trained successfully.")

    def process(self):
        files = self.get_file_paths(self.females_training_path, self.males_training_path)
        
        for file in files:
            self.total_sample += 1
            filename = os.path.basename(file)
            print("%10s %8s %1s" % ("--> TESTING", ":", filename))

            try: 
                # Extract features from audio
                vector = self.features_extractor.extract_features(file)
                
                # OPTIMIZATION: Downsample to speed up GMM fitting
                vector = vector[::5]

                # Fit a temporary GMM to this single file to get its "Supervectors"
                spk_gmm = hmm.GaussianHMM(n_components=16, covariance_type='diag', n_iter=10)
                spk_gmm.fit(vector)
                
                # Get the means of this file
                spk_vec = spk_gmm.means_ # Shape (16, 39)
                
                # Predict gender for EACH of the 16 components
                # Note: predict_classes is removed in newer Keras, using argmax instead
                predictions = np.argmax(self.model.predict(spk_vec, verbose=0), axis=-1)
                
                # COUNT VOTES
                # 0 = Female, 1 = Male
                female_votes = np.sum(predictions == 0)
                male_votes   = np.sum(predictions == 1)
                
                if male_votes > female_votes:
                    sc = 1
                    winner = "male"
                else:
                    sc = 0
                    winner = "female"
                
                # Check Expectation (Robust Logic for Windows Paths)
                if "female" in file.lower():
                    expected_gender = "female"
                elif "male" in file.lower():
                    expected_gender = "male"
                else:
                    expected_gender = "unknown"
                
                print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
                print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))
                print(f"   Votes -> Female: {female_votes} | Male: {male_votes}")
    
                if winner != expected_gender: 
                    self.error += 1
                print("----------------------------------------------------")
    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            
        # Final Accuracy
        if self.total_sample > 0:
            accuracy     = (float(self.total_sample - self.error) / float(self.total_sample)) * 100
            accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
            print(accuracy_msg)

    def get_file_paths(self, females_training_path, males_training_path):
        females = [os.path.join(females_training_path, f) for f in os.listdir(females_training_path) if f.endswith(".wav")]
        males   = [os.path.join(males_training_path, f) for f in os.listdir(males_training_path) if f.endswith(".wav")]
        return females + males

if __name__== "__main__":
    # Ensure these point to the .hmm files you saved with ModelsTrainer
    gender_identifier = GenderIdentifier(
        "TestingData/females", 
        "TestingData/males", 
        "models/females.hmm", 
        "models/males.hmm"
    )
    gender_identifier.process()