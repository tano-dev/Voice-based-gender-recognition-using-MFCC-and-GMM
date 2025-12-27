import os
import pickle
import warnings
import numpy as np
from hmmlearn import hmm
from nnCode.FeaturesExtractor import FeaturesExtractor 

warnings.filterwarnings("ignore")

class ModelsTrainer:
    def __init__(self, females_files_path, males_files_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.features_extractor    = FeaturesExtractor()

    def get_file_paths(self, females_training_path, males_training_path):
        # Join paths safely for any OS
        females = [os.path.join(females_training_path, f) for f in os.listdir(females_training_path) if f.endswith('.wav')]
        males   = [os.path.join(males_training_path, f) for f in os.listdir(males_training_path) if f.endswith('.wav')]
        return females, males

    def train_model_for_class(self, files, label_name):
        """
        Extracts features from ALL files in a list and trains ONE GMM for that class.
        Includes optimization (downsampling) to prevent 'running forever'.
        """
        all_features = np.asarray(())
        
        print(f"--> STARTING TRAINING FOR: {label_name}")
        
        count = 0
        for file in files:
            count += 1
            if count % 10 == 0:
                print(f"Processing {label_name} file {count}/{len(files)}")
            
            try:
                vector = self.features_extractor.extract_features(file)
                
                # --- OPTIMIZATION START ---
                # Take every 5th frame. Reduces data size by 80% but keeps the pattern.
                vector = vector[::5] 
                # --- OPTIMIZATION END ---
                
                if all_features.size == 0:
                    all_features = vector
                else:
                    all_features = np.vstack((all_features, vector))
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

        print(f"Total features shape for {label_name}: {all_features.shape}")

        if all_features.size > 0:
            print(f"Fitting GMM for {label_name} (this may still take a moment)...")
            # Reduced n_iter from 100 to 20 for faster training
            model_gmm = hmm.GaussianHMM(n_components=16, covariance_type='diag', n_iter=20)
            model_gmm.fit(all_features)
            return model_gmm
        else:
            print(f"No features extracted for {label_name}!")
            return None

    def process(self):
        females, males = self.get_file_paths(self.females_training_path, self.males_training_path)
        
        # 1. Train Female Model
        female_gmm = self.train_model_for_class(females, "female")
        if female_gmm:
            self.save_gmm(female_gmm, "females")
            
        # 2. Train Male Model
        male_gmm = self.train_model_for_class(males, "male")
        if male_gmm:
            self.save_gmm(male_gmm, "males")

    def save_gmm(self, gmm, name):
        """ Save the actual GMM object """
        # Create a 'models' directory if it doesn't exist to keep things clean
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.join(base_path, "models")
        if not os.path.exists(models_path):
            os.makedirs(models_path)
            
        filename = os.path.join(models_path, name + ".nn")
        
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print(f"SAVED MODEL: {filename}")

if __name__== "__main__":
    # Ensure paths are correct relative to where you run the script
    trainer = ModelsTrainer("TrainingData/females", "TrainingData/males")
    trainer.process()