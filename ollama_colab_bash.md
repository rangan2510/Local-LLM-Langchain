# Step 1: run in notebook
```
!apt-get install -qq pciutils
!curl -fsSL https://ollama.com/install.sh | sh
!curl -L -o biomistral.gguf https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q8_0.gguf?download=true
!echo "FROM ./biomistral.gguf" > Modelfile
```

## Other models

- Merge with Zephyr
    ```
    !curl -L -o biomistral.gguf https://huggingface.co/BioMistral/BioMistral-7B-Zephyr-Beta-SLERP-GGUF/resolve/main/ggml-model-Q8_0.gguf?download=true
    ```

# run the colab pro terminal. 
```
ollama serve
```

# run in notebook
```
!ollama create biomistral -f Modelfile
```

# Python setup
```
!pip install langchain langchain-community faiss-cpu langchain-openai tiktoken openpyxl pymed google-generativeai
```