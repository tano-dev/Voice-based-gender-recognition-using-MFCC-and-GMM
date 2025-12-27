import os
import pickle
import warnings
import numpy as np
import librosa
import soundfile as sf
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler # <--- QUAN TRỌNG: Để chuẩn hóa dữ liệu
from nnCode.FeaturesExtractor import FeaturesExtractor
from hmmlearn import hmm

warnings.filterwarnings("ignore")

class GenderIdentifier:

    def __init__(self, females_files_path, males_files_path, females_model_path, males_model_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.error                 = 0
        self.total_sample          = 0
        self.features_extractor    = FeaturesExtractor()
        
        # 1. Load dữ liệu "Super Vectors" từ file .nn
        print("Loading models...")
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm   = pickle.load(open(males_model_path, 'rb'))
        
        # 2. Chuẩn bị dữ liệu Train
        # Class 0 = Female, Class 1 = Male
        self.X_train = np.vstack((self.females_gmm, self.males_gmm))
        self.y_train = np.hstack((np.zeros(self.females_gmm.shape[0]), 
                                  np.ones(self.males_gmm.shape[0])))
        
        print(f"Data Shape: {self.X_train.shape}")

        # 3. Chuẩn hóa dữ liệu (StandardScaler) - QUAN TRỌNG ĐỂ TRÁNH BIAS
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        
        # 4. Thiết kế mạng Neural (Tăng số neuron và đổi sang Softmax)
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(64, input_dim=39, activation='relu')) 
        self.model.add(keras.layers.Dense(32, activation='relu'))
        self.model.add(keras.layers.Dense(2, activation='softmax')) # Softmax tốt hơn cho phân loại
        
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        
        # 5. Train model (Tăng epochs lên 50)
        print("Training Neural Network...")
        self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=16, verbose=1)

    def ffmpeg_silence_eliminator(self, input_path, output_path):
        """ Loại bỏ khoảng lặng dùng Librosa (Không cần ffmpeg.exe) """
        try:
            y, sr = librosa.load(input_path, sr=None)
            y_trimmed, _ = librosa.effects.trim(y, top_db=36)
            sf.write(output_path, y_trimmed, sr, subtype='PCM_16')
            return True
        except Exception as e:
            print(f"Silence removal error: {e}")
            return False

    def process(self):
        # Lấy danh sách file
        females = [os.path.join(self.females_training_path, f) for f in os.listdir(self.females_training_path)]
        males   = [os.path.join(self.males_training_path, f) for f in os.listdir(self.males_training_path)]
        files   = females + males

        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            # 1. Xử lý khoảng lặng (BẮT BUỘC vì model được train trên dữ liệu sạch)
            temp_path = file.split('.')[0] + "_temp.wav"
            self.ffmpeg_silence_eliminator(file, temp_path)

            try:
                # 2. Trích xuất đặc trưng từ file sạch
                vector = self.features_extractor.extract_features(temp_path)
                
                # 3. Tạo GMM tạm thời để lấy Super Vector (Means)
                spk_gmm = hmm.GaussianHMM(n_components=16)      
                spk_gmm.fit(vector)
                spk_vec = spk_gmm.means_ # Shape (16, 39)
                
                # 4. CHUẨN HÓA vector test theo scaler đã học từ tập train
                spk_vec_scaled = self.scaler.transform(spk_vec)

                # 5. Dự đoán
                # Thay predict_classes (đã cũ) bằng argmax
                predictions = self.model.predict(spk_vec_scaled) 
                predicted_classes = np.argmax(predictions, axis=1) # [0, 1, 1, 0...]
                
                # Đếm phiếu bầu (Majority Voting)
                votes_female = np.sum(predicted_classes == 0)
                votes_male   = np.sum(predicted_classes == 1)
                
                if votes_male > votes_female: winner = "male"
                else:                         winner = "female"
                
                # 6. Lấy nhãn đúng (Fix lỗi Windows Path)
                folder_name = os.path.basename(os.path.dirname(file))
                expected_gender = folder_name[:-1] # 'females' -> 'female'

                print(f"+ EXPECTATION  : {expected_gender}")
                print(f"+ IDENTIFICATION : {winner} (Votes: M={votes_male}, F={votes_female})")

                if winner != expected_gender:
                    self.error += 1
                print("----------------------------------------------------")

            except Exception as e:
                print(f"Error processing {file}: {e}")
            
            # Xóa file tạm
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Tính độ chính xác cuối cùng
        if self.total_sample > 0:
            accuracy = ((self.total_sample - self.error) / float(self.total_sample)) * 100
        else:
            accuracy = 0
        print(f"*** Accuracy = {accuracy:.3f}% ***")

if __name__== "__main__":
    # Đảm bảo đường dẫn file .nn đúng
    gender_identifier = GenderIdentifier("TestingData/females", "TestingData/males", "females.nn", "males.nn")
    gender_identifier.process()