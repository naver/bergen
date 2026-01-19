import time
import os
import glob
import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Argument parser
parser = argparse.ArgumentParser(description="Translate JSON files using a generative model with vLLM.")
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input JSON files.")
parser.add_argument("--out_dir", type=str, required=True, help="Directory to save translated JSON files.")
parser.add_argument("--lang", type=str, default="Arabic", help="Target language for translation.")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for translation.")
parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Number of GPUs to use for tensor parallelism.")
args = parser.parse_args()

print(args)

# Initialize vLLM model
model_name = "ModelSpace/GemmaX2-28-9B-v0.1"
print(f"Loading model: {model_name}")
llm = LLM(
    model=model_name,
    tensor_parallel_size=args.tensor_parallel_size,
    dtype="bfloat16",
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def truncate_prompt(prompt, tokenizer, max_len=8192):
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_len:
        print(f"⚠️ Truncating prompt from {len(tokens)} to {max_len} tokens.")
        tokens = tokens[:max_len]
    return tokenizer.decode(tokens, skip_special_tokens=True)

# File paths
input_dir = args.input_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
files = glob.glob(f"{input_dir}/*.json")
existing_files = glob.glob(f"{out_dir}/*.json")
file_names = {os.path.basename(f) for f in files}
existing_file_names = {os.path.basename(f) for f in existing_files}
remaining_files = file_names - existing_file_names
files = [input_dir + "/" + f for f in remaining_files]

print(f"Found {len(files)} files to process")

# Define sampling parameters for generation
sampling_params = SamplingParams(
    temperature=0.0,  # Use greedy decoding
    max_tokens=200,
    stop=None
)

# Aya translation prompt generator
def apply_translation_prompt(text, target_lang="Arabic"):
    return f"Translate this from English to {target_lang}:\nEnglish: {text}\n{target_lang}:"

def batch_data(files, batch_size):
    """Create batches from file paths"""
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        batch_data = []
        
        for file_path in batch_files:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            batch_data.append({
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "query": data["query"],
                "context": data["context"],
                "selected_sents": data["selected_sents"],
                "response": data["response"]
            })
        
        yield batch_data

def process_batch(batch):
    """Process a batch of data"""
    # Flatten queries and contexts for batch translation
    input_texts = []
    split_indices = []
    metadata = []
    
    for item in batch:
        input_texts.append(item["query"])
        input_texts.extend(item["context"])
        split_indices.append((len(input_texts) - 1, len(item["context"]))) # store end index of query and length of context
        metadata.append({
            "file_name": item["file_name"],
            "file_path": item["file_path"],
            "selected_sents": item["selected_sents"],
            "response": item["response"]
        })
    
    # Create prompts for translation
    prompts = [apply_translation_prompt(text, args.lang) for text in input_texts]

    # Add truncation safeguard here
    prompts = [truncate_prompt(p, tokenizer, max_len=8192 - 300) for p in prompts]
    
    # Generate translations with vLLM
    outputs = llm.generate(prompts, sampling_params)
    
    # Process results
    translations = []
    for output in outputs:
        translated_text = output.outputs[0].text.split(f"{args.lang}:")[-1].strip()
        translations.append(translated_text)
    
    return translations, split_indices, metadata

# Translation loop
start_time = time.time()
total_batches = (len(files) + args.batch_size - 1) // args.batch_size  # Ceiling division

for batch in tqdm(batch_data(files, args.batch_size), total=total_batches, desc="Translating files"):
    translations, split_indices, metadata = process_batch(batch)
    
    # Reconstruct data structure and save results
    start_idx = 0
    for i, (item_metadata, (query_end_idx, context_len)) in enumerate(zip(metadata, split_indices)):
        translated_query = translations[query_end_idx - context_len]
        translated_context = translations[query_end_idx - context_len + 1:query_end_idx + 1]
        
        out_json = {
            "query": translated_query,
            "context": translated_context,
            "selected_sents": item_metadata["selected_sents"],
            "response": item_metadata["response"]
        }
        
        out_path = os.path.join(out_dir, item_metadata["file_name"])
        with open(out_path, "w") as out_f:
            json.dump(out_json, out_f)

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
