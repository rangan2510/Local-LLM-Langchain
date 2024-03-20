curl -L -o llamafile.exe https://github.com/Mozilla-Ocho/llamafile/releases/download/0.6/llamafile-0.6
curl -L -o biomistral.gguf https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q4_K_S.gguf?download=true
chmod u+x llamafile.exe
./llamafile.exe -m .\biomistral.gguf -ngl 9999 --nobrowser