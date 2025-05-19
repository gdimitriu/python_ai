import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_arg_parser():
    parser = argparse.ArgumentParser(description='llm')
    parser.add_argument('--input-model', dest='input_model', type=str,
                        default='.', help='Input granite model')
    return parser

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    device = "cpu"
    model_path = args.input_model
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # drop device_map if running on CPU
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    model.eval()
    # change input text as desired
    chat = [
        {"role": "user",
         "content": "Please list one IBM Research laboratory located in the United States. You should only output its name and location."},
    ]
    chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # tokenize the text
    input_tokens = tokenizer(chat, return_tensors="pt").to(device)
    # generate output tokens
    output = model.generate(**input_tokens,
                            max_new_tokens=100)
    # decode output tokens into text
    output = tokenizer.batch_decode(output)
    # print output
    print(output)
