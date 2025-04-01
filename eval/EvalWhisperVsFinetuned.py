import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset
import jiwer
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

# Dataset configurations (Only Common Voice bn and en)
DATASET_CONFIGS = [
    {
        "name": "Common Voice bn",
        "path": "mozilla-foundation/common_voice_17_0",
        "config": "bn",
        "split": "test",
        "text_key": "sentence"
    },
    {
        "name": "Common Voice en",
        "path": "mozilla-foundation/common_voice_17_0",
        "config": "en",
        "split": "test",
        "text_key": "sentence"
    }
]

def load_model_and_processor(model_id):
    """Load model and processor from Hugging Face Hub or local directory."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    return model, processor

def evaluate_dataset(dataset_config, pipe, pipeline_args, sample_size=100):
    """Evaluate a single model on a dataset with streaming enabled."""
    try:
        dataset = load_dataset(
            dataset_config["path"],
            dataset_config.get("config"),
            split=dataset_config["split"],
            streaming=True
        ).shuffle(seed=42).take(sample_size)
    except Exception as e:
        print(f"Error loading {dataset_config['name']}: {str(e)}")
        return None

    predictions, references = [], []

    for sample in tqdm(dataset, desc=f"Evaluating {dataset_config['name']}"):
        try:
            audio = sample["audio"]["array"]
            if audio.size == 0 or audio.ndim != 1:
                raise ValueError("Invalid audio format")

            text = sample.get(dataset_config["text_key"], "").strip()
            if not text:
                raise ValueError("Invalid text format")

            result = pipe(audio, **pipeline_args)
            prediction = result["text"].strip()

            references.append(text)
            predictions.append(prediction)
        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            continue

    if len(references) == 0:
        return None

    wer = 100 * calculate_wer(references, predictions)
    return {
        "samples_tested": len(references),
        "WER (%)": wer
    }

def evaluate_model(model_id, dataset_configs):
    # Load model and processor
    model, processor = load_model_and_processor(model_id)
    # Create ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device="cpu",
        torch_dtype=torch.float32
    )
    pipeline_args = {
        "chunk_length_s": 30,
        "stride_length_s": (4, 2),
        "generate_kwargs": {"language": "<|en|>", "task": "transcribe"}
    }

    # Evaluate datasets
    results = []
    for dataset_config in dataset_configs:
        print(f"Evaluating on dataset: {dataset_config['name']}")
        result = evaluate_dataset(dataset_config, pipe, pipeline_args)
        if result:
            results.append({
                "model": model_id,
                "dataset": dataset_config["name"],
                **result
            })

    return results

def main():
    original_model_id = "openai/whisper-small"
    fine_tuned_model_id = "path/to/your/fine-tuned/model"  # Change this to the path of your fine-tuned model

    # Evaluate original model
    print("Evaluating original pretrained model...")
    original_results = evaluate_model(original_model_id, DATASET_CONFIGS)
    
    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    fine_tuned_results = evaluate_model(fine_tuned_model_id, DATASET_CONFIGS)

    # Print results
    print("\nOriginal Model Results:")
    for result in original_results:
        print(f"Model: {result['model']}, Dataset: {result['dataset']}, Samples Tested: {result['samples_tested']}, WER (%): {result['WER (%)']:.2f}")

    print("\nFine-Tuned Model Results:")
    for result in fine_tuned_results:
        print(f"Model: {result['model']}, Dataset: {result['dataset']}, Samples Tested: {result['samples_tested']}, WER (%): {result['WER (%)']:.2f}")

if __name__ == "__main__":
    main()
