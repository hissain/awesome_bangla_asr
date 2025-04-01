import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, DatasetDict
import jiwer
import evaluate
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Text normalization transform
wer_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords()
])

def calculate_wer(references, predictions):
    try:
        return jiwer.wer(
            references,
            predictions,
            truth_transform=wer_transform,
            hypothesis_transform=wer_transform
        )
    except:
        return float('inf')

# Model configuration for Whisper-small
MODEL_CONFIG = {
    "name": "Whisper-small",
    "model_id": "openai/whisper-small",
    "torch_dtype": torch.float32,
    "device": "cpu",
    "pipeline_args": {
        "chunk_length_s": 30,
        "stride_length_s": (4, 2),
        "generate_kwargs": {"language": "<|en|>", "task": "transcribe"}
    }
}

# Dataset configurations (Only Common Voice bn and en)
DATASET_CONFIGS = [
    {
        "name": "Common Voice bn",
        "path": "mozilla-foundation/common_voice_17_0",
        "config": "bn",
        "split": "train",
        "text_key": "sentence"
    },
    {
        "name": "Common Voice en",
        "path": "mozilla-foundation/common_voice_17_0",
        "config": "en",
        "split": "train",
        "text_key": "sentence"
    }
]

def load_model_and_processor(model_config):
    """Load model and processor from Hugging Face Hub."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_config["model_id"],
        torch_dtype=model_config.get("torch_dtype", torch.float32),
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_config["model_id"],
        trust_remote_code=True
    )
    return model, processor

def preprocess_function(examples, processor, text_key):
    audio = examples["audio"]
    text = examples[text_key]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    with processor.as_target_processor():
        labels = processor(text, return_tensors="pt").input_ids
    inputs["labels"] = labels
    return inputs

def load_and_preprocess_datasets(dataset_configs, processor):
    """Load and preprocess datasets."""
    datasets = DatasetDict()
    for config in dataset_configs:
        dataset = load_dataset(
            config["path"],
            config["config"],
            split=config["split"],
            streaming=True
        )
        dataset = dataset.map(lambda x: preprocess_function(x, processor, config["text_key"]), remove_columns=["audio", config["text_key"]])
        datasets[config["name"]] = dataset
    return datasets

def main():
    # Load model and processor
    model, processor = load_model_and_processor(MODEL_CONFIG)
    
    # Load and preprocess datasets
    datasets = load_and_preprocess_datasets(DATASET_CONFIGS, processor)
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True if torch.cuda.is_available() else False
    )

    # Define WER metric
    wer = evaluate.load("wer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        wer_score = wer.compute(predictions=pred_str, references=label_str)
        return {"wer": wer_score}

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets[DATASET_CONFIGS[0]["name"]],
        eval_dataset=datasets[DATASET_CONFIGS[1]["name"]],
        data_collator=lambda data: {"input_ids": torch.stack([f["input_ids"] for f in data]),
                                    "attention_mask": torch.stack([f["attention_mask"] for f in data]),
                                    "labels": torch.stack([f["labels"] for f in data])},
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

if __name__ == "__main__":
    main()
