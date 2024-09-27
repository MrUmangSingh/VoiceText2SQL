import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd
import numpy as np

def initialize_whisper():
    print("Initializing Whisper model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
    return processor, model

def transcribe_audio(processor, model, duration=5, sample_rate=16000):
    # Record audio
    print("Recording audio...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")

    # Process audio
    input_features = processor(np.squeeze(audio), sampling_rate=sample_rate, return_tensors="pt")

    # Create proper attention mask
    attention_mask = torch.ones_like(input_features.input_features)
    attention_mask = attention_mask.to(torch.long)

    # Set forced_decoder_ids for English
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    # Transcribe
    with torch.no_grad():
        predicted_ids = model.generate(
        input_features.input_features, 
        forced_decoder_ids=forced_decoder_ids,
        attention_mask=attention_mask
    )
    # Return transcription
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
