import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, Audio
from transformers import (
    VitsTokenizer, 
    VitsConfig, 
    VitsModel,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    AutoProcessor
)
import gradio as gr
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. LOAD AND PREPROCESS MOZILLA COMMON VOICE BENGALI DATASET
def prepare_dataset():
    print("Loading Mozilla Common Voice Bengali dataset...")
    # Load dataset from HuggingFace (you may need to log in)
    dataset = load_dataset("mozilla-foundation/common_voice_11_0", "bn", split="train+validation")
    
    # Filter out samples with no text or audio
    dataset = dataset.filter(lambda x: x["sentence"] is not None and len(x["sentence"]) > 0)
    
    # Convert to audio format that matches model requirements
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Split into train and validation
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
    
    # Convert to pandas DataFrames for easier manipulation
    train_df = pd.DataFrame(train_dataset)
    val_df = pd.DataFrame(val_dataset)
    
    print(f"Train dataset size: {len(train_df)}")
    print(f"Validation dataset size: {len(val_df)}")
    
    return train_dataset, val_dataset, train_df, val_df

# 2. MODEL SELECTION AND PREPARATION
def load_model_and_processor(model_name="facebook/mms-tts-ben"):
    """
    Load the pre-trained MMS-TTS model for Bengali.
    If not available, we'll use a small model and adapt it.
    """
    try:
        print(f"Loading model: {model_name}")
        tokenizer = VitsTokenizer.from_pretrained(model_name)
        config = VitsConfig.from_pretrained(model_name)
        model = VitsModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        return model, processor, tokenizer, config
    except Exception as e:
        print(f"Error loading specific model: {e}")
        print("Loading a general small model and will adapt for Bengali...")
        # Fall back to a small model that we can adapt
        model_name = "facebook/mms-tts-eng"
        tokenizer = VitsTokenizer.from_pretrained(model_name)
        config = VitsConfig.from_pretrained(model_name)
        
        # Adjust config for Bengali if needed
        # This may require customizing the phoneme set or other language-specific settings
        
        model = VitsModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        return model, processor, tokenizer, config

# 3. DATA PREPARATION FOR TRAINING
@dataclass
class TTSDataCollator:
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        labels = [{"input_values": feature["audio"]["array"]} for feature in features]
        
        batch = self.processor.pad(input_ids=input_ids, labels=labels, return_tensors="pt")
        
        # Get audio lengths
        batch["labels_attention_mask"] = torch.tensor(
            [len(feature["audio"]["array"]) for feature in features]
        )
        
        return batch

def prepare_features(examples, processor, tokenizer):
    # Tokenize text
    inputs = tokenizer(examples["sentence"], return_tensors="pt", padding=True)
    
    # Process audio (ensure sampling rate is correct)
    audio = examples["audio"]["array"]
    sampling_rate = examples["audio"]["sampling_rate"]
    
    if sampling_rate != 16000:
        # Resample audio if needed
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    
    # Return processed features
    return {
        "input_ids": inputs["input_ids"][0],
        "audio": {"array": audio, "sampling_rate": 16000},
        "sentence": examples["sentence"]
    }

def prepare_datasets_for_training(train_dataset, val_dataset, processor, tokenizer):
    # Apply preprocessing to datasets
    train_dataset = train_dataset.map(
        lambda x: prepare_features(x, processor, tokenizer),
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda x: prepare_features(x, processor, tokenizer),
        remove_columns=val_dataset.column_names
    )
    
    return train_dataset, val_dataset

# 4. TRAINING SETUP
def setup_training(model, train_dataset, val_dataset, processor, output_dir="./tts_bengali_model"):
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,  # Adjust based on your GPU memory
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=10,  # You might need more for good results
        warmup_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=3,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        streaming=True,  # For space efficiency as requested
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,  # Set to True if you want to upload to HuggingFace
    )
    
    data_collator = TTSDataCollator(processor=processor)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    return trainer

# 5. FINE-TUNING PROCESS
def finetune_model(trainer):
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete!")
    
    # Save the fine-tuned model
    trainer.save_model("./final_tts_bengali_model")
    print("Model saved to ./final_tts_bengali_model")
    
    return trainer

# 6. EVALUATION AND COMPARISON
def evaluate_models(original_model, finetuned_model, processor, tokenizer, test_sentences, output_dir="./evaluation"):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    original_model = original_model.to(device)
    finetuned_model = finetuned_model.to(device)
    
    results = []
    
    for i, sentence in enumerate(test_sentences):
        print(f"Processing test sentence {i+1}/{len(test_sentences)}")
        
        # Tokenize the sentence
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        
        # Generate speech with original model
        with torch.no_grad():
            original_output = original_model(
                input_ids=inputs["input_ids"],
                return_dict=True,
                output_type="waveform",
            )
        
        original_waveform = original_output.waveform.squeeze().cpu().numpy()
        original_path = os.path.join(output_dir, f"original_sample_{i}.wav")
        sf.write(original_path, original_waveform, 16000)
        
        # Generate speech with fine-tuned model
        with torch.no_grad():
            finetuned_output = finetuned_model(
                input_ids=inputs["input_ids"],
                return_dict=True,
                output_type="waveform",
            )
        
        finetuned_waveform = finetuned_output.waveform.squeeze().cpu().numpy()
        finetuned_path = os.path.join(output_dir, f"finetuned_sample_{i}.wav")
        sf.write(finetuned_path, finetuned_waveform, 16000)
        
        # Calculate basic audio metrics
        original_duration = len(original_waveform) / 16000
        finetuned_duration = len(finetuned_waveform) / 16000
        
        # Calculate mel spectrograms for comparison
        original_mel = librosa.feature.melspectrogram(y=original_waveform, sr=16000)
        finetuned_mel = librosa.feature.melspectrogram(y=finetuned_waveform, sr=16000)
        
        # Calculate spectral difference
        min_length = min(original_mel.shape[1], finetuned_mel.shape[1])
        spec_diff = np.mean(np.abs(original_mel[:, :min_length] - finetuned_mel[:, :min_length]))
        
        results.append({
            "sentence": sentence,
            "original_path": original_path,
            "finetuned_path": finetuned_path,
            "original_duration": original_duration,
            "finetuned_duration": finetuned_duration,
            "spectral_difference": spec_diff
        })
    
    # Summarize results
    results_df = pd.DataFrame(results)
    print("Evaluation Results:")
    print(results_df[["original_duration", "finetuned_duration", "spectral_difference"]].describe())
    
    # Visualize spectrograms for the first sample
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.power_to_db(
        librosa.feature.melspectrogram(y=original_waveform, sr=16000), ref=np.max), 
        y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Model Mel-frequency spectrogram')
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.power_to_db(
        librosa.feature.melspectrogram(y=finetuned_waveform, sr=16000), ref=np.max), 
        y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Fine-tuned Model Mel-frequency spectrogram')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spectrogram_comparison.png"))
    
    return results_df

# 7. CREATE DEMO INTERFACE
def create_demo(finetuned_model, tokenizer, processor):
    def synthesize_speech(text):
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # Generate speech
        with torch.no_grad():
            output = finetuned_model(
                input_ids=inputs["input_ids"],
                return_dict=True,
                output_type="waveform",
            )
        
        waveform = output.waveform.squeeze().cpu().numpy()
        
        # Save as temporary file
        temp_path = "temp_audio.wav"
        sf.write(temp_path, waveform, 16000)
        
        return temp_path
    
    # Create Gradio interface
    demo = gr.Interface(
        fn=synthesize_speech,
        inputs=gr.Textbox(lines=3, placeholder="Enter Bengali text here..."),
        outputs=gr.Audio(type="filepath"),
        title="Bengali Text-to-Speech",
        description="Enter Bengali text and generate speech using a fine-tuned model."
    )
    
    return demo

# 8. COMPRESS MODEL FOR ANDROID
def optimize_for_android(model_dir, output_dir="./android_model"):
    """
    Convert and optimize the model for Android deployment.
    This uses ONNX conversion and quantization for size reduction.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx
    
    # Load the saved model
    model = VitsModel.from_pretrained(model_dir)
    tokenizer = VitsTokenizer.from_pretrained(model_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate example input for tracing
    sample_text = "এটি একটি নমুনা বাক্য।"  # "This is a sample sentence." in Bengali
    inputs = tokenizer(sample_text, return_tensors="pt")
    
    # Export to ONNX
    print("Converting model to ONNX format...")
    torch.onnx.export(
        model,
        (inputs["input_ids"],),
        f"{output_dir}/model.onnx",
        input_names=["input_ids"],
        output_names=["waveform"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "waveform": {0: "batch_size", 1: "audio_length"}
        },
        opset_version=13
    )
    
    # Optimize ONNX model
    print("Optimizing ONNX model...")
    onnx_model = onnx.load(f"{output_dir}/model.onnx")
    
    # Apply quantization to reduce size
    print("Quantizing model to reduce size...")
    quantized_model_path = f"{output_dir}/model_quantized.onnx"
    quantize_dynamic(
        f"{output_dir}/model.onnx",
        quantized_model_path,
        weight_type=QuantType.QUInt8
    )
    
    # Copy tokenizer and config
    tokenizer.save_pretrained(output_dir)
    
    print(f"Optimized model saved to {output_dir}")
    print(f"Original ONNX model size: {os.path.getsize(f'{output_dir}/model.onnx') / (1024 * 1024):.2f} MB")
    print(f"Quantized ONNX model size: {os.path.getsize(quantized_model_path) / (1024 * 1024):.2f} MB")
    
    return output_dir

# 9. MAIN EXECUTION FUNCTION
def main():
    # Prepare dataset
    train_dataset, val_dataset, _, _ = prepare_dataset()
    
    # Load model
    model, processor, tokenizer, config = load_model_and_processor()
    
    # Prepare datasets for training
    train_dataset, val_dataset = prepare_datasets_for_training(
        train_dataset, val_dataset, processor, tokenizer
    )
    
    # Setup training
    trainer = setup_training(model, train_dataset, val_dataset, processor)
    
    # Fine-tune model
    trainer = finetune_model(trainer)
    
    # Evaluate and compare
    original_model, _, _, _ = load_model_and_processor()  # Reload original for comparison
    finetuned_model = VitsModel.from_pretrained("./final_tts_bengali_model")
    
    test_sentences = [
        "আমি বাংলায় কথা বলি।",  # "I speak Bengali."
        "স্বাধীনতা আমাদের অধিকার।",  # "Freedom is our right."
        "বাংলাদেশ একটি সুন্দর দেশ।"  # "Bangladesh is a beautiful country."
    ]
    
    results = evaluate_models(original_model, finetuned_model, processor, tokenizer, test_sentences)
    
    # Optimize for Android
    android_model_dir = optimize_for_android("./final_tts_bengali_model")
    
    # Create demo
    demo = create_demo(finetuned_model, tokenizer, processor)
    demo.launch()
    
    print("Process complete!")

if __name__ == "__main__":
    main()