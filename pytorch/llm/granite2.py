from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "ibm-granite/granite-3.0-3b-a800m-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the prompt
prompt = "Hello, how are you today?"

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
output = model.generate(input_ids, max_length=1000, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)