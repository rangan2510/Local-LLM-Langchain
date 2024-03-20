curl -fsSL https://ollama.com/install.sh | sh
curl -L -o biomistral.gguf https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q8_0.gguf?download=true
echo "FROM ./biomistral.gguf" > Modelfile
ollama create biomistral -f Modelfile

# run the above lines with '!' in notebook.
# run the line below in the terminal. 

ollama serve