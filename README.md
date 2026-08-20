# Agent Writer

conda create -n agent-writer python=3.11 -y
conda activate agent-writer
pip install -r requirements.txt
python app.py

ollama list
ollama pull llama3.1:8b
