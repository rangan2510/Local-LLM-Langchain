# Step 1: run in notebook
```
!apt-get install pciutils
!curl -fsSL https://ollama.com/install.sh | sh
curl -L -o biomistral.gguf https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q8_0.gguf?download=true
!echo "FROM ./biomistral.gguf" > Modelfile
```

# run the colab pro terminal. 
```
ollama serve
```
# run in notebook
```
!ollama create biomistral -f Modelfile
```