import requests
import gradio as gr
import json
import threading
import time

# DeepSeek API URL (use 127.0.0.1 to avoid any localhost resolution delays)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Legal Document Templates
LEGAL_TEMPLATES = {
    "rental agreement": "Generate a rental agreement between {party1} (tenant) and {party2} (landlord) for {duration} months.",
    "employment contract": "Generate an employment contract between {party1} (employee) and {party2} (employer) with a salary of {salary} per year.",
    "business partnership agreement": "Draft a business partnership agreement between {party1} and {party2}, defining responsibilities and profit-sharing terms.",
    "nda": "Generate a non-disclosure agreement (NDA) between {party1} and {party2} to protect confidential business information."
}

def generate_legal_document(doc_type, party1, party2, duration="", salary=""):
    """
    Uses DeepSeek AI to generate legal contracts.
    """
    # Normalize selection from UI (e.g., "NDA" -> "nda", "Rental Agreement" -> "rental agreement")
    doc_key = (doc_type or "").strip().lower()
    if doc_key not in LEGAL_TEMPLATES:
        return "Invalid document type. Please choose from rental agreement, employment contract, business partnership agreement, or NDA."

    prompt = LEGAL_TEMPLATES[doc_key].format(party1=party1, party2=party2, duration=duration, salary=salary)

    payload = {
        "model": "deepseek-r1",
        "prompt": prompt,
        # keep the model in memory for faster subsequent calls
        "keep_alive": "5m",
        # request non-streamed full response to keep UI simple for now
        "stream": False
    }
    
    try:
        # First run can take >60s while the model cold-loads; allow more time
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    except requests.RequestException as e:
        return f"Error connecting to AI server: {e}. Ensure Ollama is running (ollama serve) and the model is pulled (ollama pull deepseek-r1)."
    
    if response.status_code == 200:
        return response.json().get("response", "No document generated.")
    else:
        return f"Error: {response.text}"


def generate_legal_document_stream(doc_type, party1, party2, duration="", salary=""):
    """Stream partial output from Ollama so the UI updates incrementally."""
    # Normalize selection
    doc_key = (doc_type or "").strip().lower()
    if doc_key not in LEGAL_TEMPLATES:
        yield "Invalid document type. Please choose from rental agreement, employment contract, business partnership agreement, or NDA."
        return

    prompt = LEGAL_TEMPLATES[doc_key].format(party1=party1, party2=party2, duration=duration, salary=salary)

    payload = {
        "model": "deepseek-r1",
        "prompt": prompt,
        "keep_alive": "5m",
        "stream": True,
    }

    accumulated = ""
    try:
        # Use a reasonable connect timeout; streaming reads as chunks arrive
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=(10, 600)) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # Some runtimes may send non-JSON keepalives; skip
                    continue
                if "response" in data:
                    accumulated += data["response"]
                    # yield progressively so Gradio updates the textbox
                    yield accumulated
                if data.get("done"):
                    break
    except requests.RequestException as e:
        yield f"Error connecting to AI server: {e}. Ensure Ollama is running (ollama serve) and the model is pulled (ollama pull deepseek-r1)."
        return

# Create Gradio interface
interface = gr.Interface(
    fn=generate_legal_document_stream,
    inputs=[
        gr.Radio(["Rental Agreement", "Employment Contract", "Business Partnership Agreement", "NDA"], label="Document Type"),
        gr.Textbox(label="Party 1 Name"),
        gr.Textbox(label="Party 2 Name"),
        gr.Textbox(label="Duration (if applicable, in months)", placeholder="e.g., 12"),
        gr.Textbox(label="Salary (if applicable, per year)", placeholder="e.g., $50,000"),
    ],
    outputs=gr.Textbox(lines=20, label="Generated Legal Document"),
    title="AI-Powered Legal Assistant",
    description="Select a document type, enter party names, and generate a professional legal contract."
)


def _warmup_model():
    """Warm up the model so the first real request is fast."""
    try:
        payload = {
            "model": "deepseek-r1",
            "prompt": "Warmup.",
            "keep_alive": "10m",
            "stream": False,
        }
        # Give the warmup up to 5 minutes to avoid blocking startup
        requests.post(OLLAMA_URL, json=payload, timeout=300)
    except Exception:
        # Best-effort warmup; ignore errors
        pass

# Launch the web app
if __name__ == "__main__":
    # Preload the model in the background to avoid first-request delays
    threading.Thread(target=_warmup_model, daemon=True, name="Warmup").start()
    interface.launch()



# # Test Legal Assistant
# if __name__ == "__main__":
#     print("### AI-Generated Contract ###")
#     print(generate_legal_document("rental agreement", "John Doe", "Jane Smith", duration="12"))



