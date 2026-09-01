from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
import os

# Force Hugging Face to accept remote code without running the timer
os.environ["HF_TRUST_REMOTE_CODE"] = "1" 

MODEL_ID = 'AxiomicLabs/GPT-X2.5-135M'

# Initial embedding model and llm model 
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map = "auto"
    )

# Initial fast api
app = FastAPI()

class Request(BaseModel):
    text: str

@app.get("/health")
def check_health():
    return "Good"

@app.post("/chat")
def chat(request: Request ):

    user_request = request.text

    tokens = tokenizer(
        user_request,
        return_tensors = "pt"               
        ).to(model.device)
    
    outputs = model.generate(
        **tokens,
        max_new_tokens = 100
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens= True
        )

    return {"response": response}