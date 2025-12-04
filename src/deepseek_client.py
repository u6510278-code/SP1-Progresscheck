
import requests
from typing import List
from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

class DeepSeekClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.base_url = base_url or DEEPSEEK_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",    
            "X-Title": "notebooklm-clone",         
        })


    # embeddings 
    def embed(self, texts: List[str], model: str = "deepseek-embed") -> List[list[float]]:
        url = f"{self.base_url}/v1/embeddings"   
        payload = {"model": model, "input": texts}
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

       
    def chat(self, messages, model: str = "openrouter/auto", temperature: float = 0.2) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,        
            "messages": messages,
            "temperature": temperature,
        }
        resp = self.session.post(url, json=payload)

        if not resp.ok:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

        data = resp.json()
        return data["choices"][0]["message"]["content"]
