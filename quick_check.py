import os
from opentslm import OpenTSLM

os.makedirs("models", exist_ok=True)
m = OpenTSLM.load_pretrained(
    "OpenTSLM/llama-3.2-1b-har-flamingo", device="cpu")

m.store_to_file("models/har_pretrained.pt")
print("Done!")
