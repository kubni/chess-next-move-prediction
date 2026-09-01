# Predvidjanje sledeceg poteza u sahu

Predvidjamo **koji potez ce covek odigrati** u datoj poziciji - ne koji je
potez najbolji. Model uci ljudsko ponasanje iz partija sa lichess-a, pa se
pored pet pristupa nad istima podacima i istom metrikom


Projekat iz Masinskog ucenja, Matematicki fakultet
Autori: Jovan Skoric 1030/2024, Nikola Kuburovic

## Okruzenje

Python 3.12. Sve zavisnosti su u `requirements.txt`

```bash
python3 -m venv .venv
source .venv/bin/activate

# sa NVIDIA GPU-om (podrazumevani torch sa PyPI-ja je CUDA build):
pip install -r requirements.txt

# bez GPU-a:
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

python -m ipykernel install --user --name chessml --display-name "chessml"