# Predvidjanje sledeceg poteza u sahu

Seminarski projekat iz Masinskog ucenja, MATF

## Opis projekta

Zadatak je predvidjanje poteza koji ce igrac odigrati u datoj poziciji, formulisan
kao klasifikacija u 1968 geometrijski mogućih UCI poteza. Poredi se cetiri pristupa
na istom skupu, istoj podeli i istoj metrici: logisticka regresija, MLP i CNN
koji vide tablu, i transformer koji vidi samo niz odigranih poteza.

Metrike su top-1, top-3 i top-5 tačnost, računate i sa maskom legalnih poteza i
bez nje, uz udeo nelegalnih predloga. Rezultati su razlozeni po fazi igre i po
rejtingu igrača.

## Skup podataka

Izvor su Lichess PGN arhive partija sa standardnim pravilima. Zadržavaju se samo
rangirane partije sa regularnim zavrsetkom, klasicnim tempom i poznatim rejtingom
oba igraca.

Partije se pretvaraju u `.npy` nizove fiksne širine (`chessml/data/build_cache.py`),
jer nasumičan pristup milionima pozicija tokom treninga mora biti O(1). Jedna
pozicija je red od 70 brojeva: 64 polja sa kodom figure, 4 prava na rokadu, strana
na potezu, i en passant polje. Sve pozicije su ogledane na belog, pa model uvek
uci iz perspektive strane koja je na potezu.

Uz svaku poziciju ide odigrani potez (oznaka) i `[game_id, ply, elo_bucket]`.
Paralelno se cuva i sekvencijalni pogled — svi potezi svih partija nadovezani —
koji koristi transformer.

Struktura skupa, balansiranost oznaka i osnovna svojstva analizirani su u
`01_podaci_i_pretprocesiranje.ipynb` i `02_eksplorativna_analiza.ipynb`.

### Podela na skupove

Deli se **po partiji, nikad po poziciji**. Svaka 20. partija (5%) ide u
validaciju, ostatak u trening. Test skup se gradi iz **arhive drugog meseca**,
pa nema preklapanja ni u partijama ni u igracima.

Podela po poziciji bi pustila pozicije iste partije i u trening i u validaciju —
one dele istoriju poteza i najveci deo table, pa bi model prepoznavao partiju
umesto da uci sah, a rezultat bi bio lazno visok.

## Okruzenje

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pokretanje

```bash
# kes iz PGN arhive (jedan mesec za trening, drugi za test)
python -m chessml.data.build_cache -p data/lichess_2024-01.pgn -s trainval -g 20000
python -m chessml.data.build_cache -p data/lichess_2024-02.pgn -s test -g 5000

# podela na trening i validaciju
python -m chessml.data.train_val_sep

# modeli
python -m chessml.models.baselines
python -m chessml.models.mlp
python -m chessml.models.cnn
python -m chessml.models.transformer
```

Sveske se pregledaju redom po prefiksu, 01 do 07. Poređenje svih modela i
zakljucci su u `06_finalna_evaluacija_i_poredjenje.ipynb`, a `07_demo.ipynb`
pokazuje predviđanje na konkretnoj partiji.

Kes se ne komituje jer se ponovo gradi iz PGN-a. Istrenirani modeli su u
`artifacts/models/`, metrike u `artifacts/metrics/`, logovi treninga u `logs/`.

## Literatura

- Lichess Open Database — https://database.lichess.org (izvor PGN arhiva, CC0)
- `python-chess` dokumentacija — https://python-chess.readthedocs.io

## Tim

- Jovan Skoric (`Skora01`)
- Nikola Kuburovic (`kubni`)
