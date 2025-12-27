import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib
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
                # vector = vector[::5] 
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

            
            # FIXED: Added min_covar to prevent numerical instability
            # FIXED: Increased n_iter to 50 or 100 for better convergence
            model_gmm = hmm.GaussianHMM(
                n_components=16, 
                covariance_type='diag', 
                n_iter=50,             # Increased from 20
                min_covar=0.01,       # Prevents variance from hitting 0
                verbose=True
            )
            
            model_gmm.fit(all_features)
            
            return model_gmm, model_gmm.monitor_.history
        else:
            print(f"No features extracted for {label_name}!")
            return None, []

    def plot_training_history(self, female_hist, male_hist):
        """
        Plots the Log-Likelihood convergence for both models.
        """
        
        plt.figure(figsize=(12, 5))

        # Plot Female Data
        plt.subplot(1, 2, 1)
        if female_hist:
            plt.plot(female_hist, label='Log-Likelihood', color='purple')
            plt.title('Female Model Convergence')
            plt.xlabel('Iterations')
            plt.ylabel('Log-Likelihood')
            plt.grid(True)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No Data', ha='center')

        # Plot Male Data
        plt.subplot(1, 2, 2)
        if male_hist:
            plt.plot(male_hist, label='Log-Likelihood', color='orange')
            plt.title('Male Model Convergence')
            plt.xlabel('Iterations')
            plt.grid(True)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No Data', ha='center')

        plt.tight_layout()
        plt.show()

    def process(self):
        females, males = self.get_file_paths(self.females_training_path, self.males_training_path)
        
        # 1. Train Female Model
        female_gmm, female_hist = self.train_model_for_class(females, "female")
        if female_gmm:
            self.save_gmm(female_gmm, "females")
            
        # 2. Train Male Model
        male_gmm, male_hist = self.train_model_for_class(males, "male")
        if male_gmm:
            self.save_gmm(male_gmm, "males")
            
        # 3. Show Graph
        print("Displaying training graph...")
        self.plot_training_history(female_hist, male_hist)

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
    if os.path.exists("TrainingData/females") and os.path.exists("TrainingData/males"):
        trainer = ModelsTrainer("TrainingData/females", "TrainingData/males")
        trainer.process()
    else:
        print("Error: 'TrainingData' directory not found.")