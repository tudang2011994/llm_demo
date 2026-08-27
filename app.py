from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel

MODEL_ID = 'AxiomicLabs/GPT-X2.5-135M'

# Initial embedding model and llm model 
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map = "auto"
    )

# Initial fast api
app = FastAPI()

class Request(BaseModel):
    text: str

@app.post("/chat")
def chat(request: Request ):

    user_request = request.text

    tokens = tokenizer(
        user_request,
        return_tensor = "pt"               
        ).to(model.device)
    
    outputs = model.generate(
        **tokens,
        max_new_token = 100
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens= "true"
        )

    return {"response": response}