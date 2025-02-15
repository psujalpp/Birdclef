import os
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import whisper
import soundfile as sf
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score
import ChatModels

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_PATH = "Birdclef_GLOBAL_6K_V2.4_Model_FP32.tflite"
LABELS_PATH = "Birdclef_GLOBAL_6K_V2.4_Labels.txt"
AUDIO_FILE = "RTH.mp3"
OUTPUT_DIR = "Output/"
AUGMENTED_DIR = os.path.join(OUTPUT_DIR, "AugmentedSpectrograms/")
TRAIN_DATA_DIR = "TrainData/"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUGMENTED_DIR, exist_ok=True)
os.makedirs(TRAIN_DATA_DIR, exist_ok=True)


def save_log_mel_spectrogram(audio, sr, output_png, output_wav=None):
    try:
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel_spec, sr=sr, hop_length=512, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.savefig(output_png)
        plt.close()

        if output_wav:  
            sf.write(output_wav, audio.astype(np.float32), sr)

    except Exception as e:
        print(f"Error saving spectrogram to {output_png}: {e}")

def load_audio(file_path, sr=48000, target_length=144000):
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")
        else:
            y = y[:target_length]
        return np.array(y, dtype=np.float32), sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None

def prepare_training_data():
    for file in os.listdir("Bird_Audio_Folder/"):
        if file.endswith(".mp3") or file.endswith(".wav"):
            file_path = os.path.join("Bird_Audio_Folder/", file)
            audio_data, sr = load_audio(file_path)  
            if audio_data is not None:
                output_path = os.path.join(TRAIN_DATA_DIR, f"{file}.png")
                save_log_mel_spectrogram(audio_data, sr, output_path)
                print(f"Saved spectrogram: {output_path}")

prepare_training_data()  

def train_cnn_model(data_dir):
    try:
        datagen = ImageDataGenerator(
            rescale=1./255, 
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )

        train_gen = datagen.flow_from_directory(
            data_dir, target_size=(224, 224), batch_size=32, class_mode="categorical", subset="training"
        )
        val_gen = datagen.flow_from_directory(
            data_dir, target_size=(224, 224), batch_size=32, class_mode="categorical", subset="validation"
        )

        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        
        
        for layer in base_model.layers[-50:]:
            layer.trainable = True
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(len(train_gen.class_indices), activation="softmax")
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        history = model.fit(train_gen, validation_data=val_gen, epochs=15)

        
        model.save(os.path.join(OUTPUT_DIR, "cnn_bird_model.h5"))

        return {
            "accuracy": max(history.history["accuracy"]),
            "loss": min(history.history["loss"])
        }

    except Exception as e:
        print(f"CNN training failed: {e}")
        return {"error": str(e)}


try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    EXPECTED_SAMPLES = input_shape[1]
except Exception as e:
    print(f"Error loading TFlow model: {e}")
    exit(1)

def load_labels(labels_path):
    try:
        with open(labels_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Labels file not found: {labels_path}")
        return []

bird_labels = load_labels(LABELS_PATH)

def transcribe_audio(audio_path):
    try:
        model = whisper.load_model("base")
        return model.transcribe(audio_path)["text"]
    except Exception as e:
        print(f"Error in transcription: {e}")
        return ""

def compute_f1_score(true_text, pred_text):
    try:
        return f1_score(true_text.split(), pred_text.split(), average="macro")
    except Exception as e:
        print(f"Error computing F1 score: {e}")
        return None

def predict_bird_tflite(audio_data):
    try:
        input_tensor_index = input_details[0]['index']
        output_tensor_index = output_details[0]['index']

        if len(audio_data) != EXPECTED_SAMPLES:
            audio_data = np.pad(audio_data, (0, max(0, EXPECTED_SAMPLES - len(audio_data))), mode="constant")

        audio_data = np.array(audio_data, dtype=np.float32)

        interpreter.set_tensor(input_tensor_index, np.expand_dims(audio_data, axis=0))
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_tensor_index)
        return predictions.flatten()
    except Exception as e:
        print(f"Error in TFlow prediction: {e}")
        return None

def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"Audio file {AUDIO_FILE} not found!")
        return

    audio_data, sr = load_audio(AUDIO_FILE)
    if audio_data is None:
        return

    spectrogram_path = os.path.join(OUTPUT_DIR, "original_spectrogram.png")
    temp_wav_path = os.path.join(OUTPUT_DIR, "temp_audio.wav")
    save_log_mel_spectrogram(audio_data, sr, spectrogram_path, temp_wav_path)

    transcription = transcribe_audio(temp_wav_path)
    predictions = predict_bird_tflite(audio_data)
    predicted_idx = np.argmax(predictions) if predictions is not None else -1
    predicted_species = bird_labels[predicted_idx] if predicted_idx < len(bird_labels) else "Unknown"

    whisper_f1_segmented_scores = [compute_f1_score("ground_truth_text", transcription)]
    valid_whisper_f1_scores = [score for score in whisper_f1_segmented_scores if score is not None]
    f1_whisper_segmented = np.mean(valid_whisper_f1_scores) if valid_whisper_f1_scores else None

    print(f"Predicted Bird: {predicted_species}")

    
    cnn_results = train_cnn_model(TRAIN_DATA_DIR)
    cnn_accuracy = cnn_results.get("accuracy", -1)
    cnn_loss = cnn_results.get("loss", -1)

    results = {
        "audio_file": AUDIO_FILE,
        "predicted_species": predicted_species if predicted_species else "Unknown",
        "transcription": transcription if transcription else "No transcription available",
        "whisper_f1_score": f1_whisper_segmented if f1_whisper_segmented is not None else -1,
        "spectrogram_image": spectrogram_path if os.path.exists(spectrogram_path) else "Not generated",
        "temp_audio_wav": temp_wav_path if os.path.exists(temp_wav_path) else "Not generated",
        "cnn_training": {
            "accuracy": cnn_accuracy if cnn_accuracy is not None else -1,
            "loss": cnn_loss if cnn_loss is not None else -1
        }
    }

    
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print("Results saved successfully.")

    try:
        ChatModels.META_PROMPT(str(predicted_species))
    except Exception as e:
        print(f"Error in ChatModels: {e}")

if __name__ == "__main__":
    main()
