---
language: en
license: apache-2.0
tags:
  - audio
  - speech
  - emotion-recognition
  - wav2vec2
datasets:
  - RAVDESS
model-index:
  - name: wav2vec2-emotion-recognition-for-ravdness
    results:
      - task:
          name: Speech Emotion Recognition
          type: audio-classification
        metrics:
          - name: Training Accuracy
            value: 0.9427  # Training accuracy from last epoch
          - name: Validation Accuracy
            value: 0.9427 
          - name: Training Loss
            value: 2.38  
          - name: Validation Loss
            value: 0.26  
---

# Speech Emotion Recognition Model

This model is fine-tuned for speech emotion recognition. It can detect emotions such as happiness, sadness, anger, fear, disgust, etc. in speech.

## Model Details

- Model type: Fine-tuned Wav2Vec2
- Base model: lighteternal/wav2vec2-large-xlsr-53-english
- Training data: RAVDESS dataset
- Supported emotions: anger, disgust, fear, happiness, sadness

## Usage

```python
from transformers import Wav2Vec2Processor, AutoModelForAudioClassification
import torchaudio
import torch

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained("usamakenway/wav2vec2-large-xlsr-53-english-ravdess")
model = AutoModelForAudioClassification.from_pretrained("usamakenway/wav2vec2-large-xlsr-53-english-ravdess")

# Function to predict emotion from audio file
def predict_emotion(audio_path):
    # Load audio
    speech_array, sampling_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech = resampler(speech_array).squeeze().numpy()
    else:
        speech = speech_array.squeeze().numpy()
    
    # Process audio
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        predicted_emotion = model.config.id2label[predicted_class_id]
    
    return predicted_emotion

# Example usage
emotion = predict_emotion("path/to/audio.wav")
print(f"Detected emotion: sadness")
```

## Training Details

The model was trained for 8 epochs using the following parameters:
- Learning rate: 1e-4
- Batch size: 20
- Gradient accumulation steps: 4

## Limitations

This model works best with clear speech recordings in quiet environments. Performance may vary with different accents, languages, or noisy backgrounds.
