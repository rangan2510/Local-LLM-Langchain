# Run in Windows Powershell
For local execution only. 
```
winget install Ollama.ollama
curl -L -o biomistral.gguf https://huggingface.co/BioMistral/BioMistral-7B-GGUF/resolve/main/ggml-model-Q4_K_S.gguf?download=true
echo "FROM ./biomistral.gguf" > Modelfile
ollama create biomistral -f Modelfile
ollama serve
```