# Txukun CLI

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://docs.astral.sh/uv/)

Euskarazko testuen **maiuskulak, puntuazioa eta ortografia** zuzentzeko komando-lerroko tresna.

- 🧠 **Cap+Punct**: `itzune/txukun-cap-punct-eu` ONNX int8 eredu kuantizatua (~77 MB, `optimum[onnxruntime]`)
- 🔍 **Ortografia**: 160.000 hitzeko euskal hiztegi propioa (Xuxen + corpus maiztasunak)
- ⚡ **Lehen zuzenketa automatikoa**: akats ortografikoak automatikoki zuzentzen ditu

> **Itzune** kolektiboaren proiektua — euskarazko AI tresna libre eta pribatuak.

---

## Instalazioa

```bash
git clone https://github.com/itzune/txukun-cli.git
cd txukun-cli
uv sync
```

`uv sync` komandoak beharrezko mendekotasun guztiak instalatuko ditu:
- `optimum[onnxruntime]` — ONNX eredua exekutatzeko (CPU, ~77 MB deskarga)
- `click` — CLI interfazerako

---

## Erabilera

### Oinarrizkoa

```bash
uv run python txukun.py "ser gertatu da hemen"
# → Ser gertatu da hemen.
```

### Ortografia zuzenketarekin

```bash
uv run python txukun.py --spell "ser gertatu da hemen"
# → zer gertatu da hemen.
```

### Fitxategitik irakurri

```bash
uv run python txukun.py --file input.txt --output zuzendua.txt
```

### Stdin bidez (pipe)

```bash
cat testua.txt | uv run python txukun.py --stdin
echo "kaixo mundua" | uv run python txukun.py --stdin
```

### Aukerak

| Aukera | Deskribapena |
|---|---|
| `TEXT` | Zuzendu beharreko testua (lehen parametroa) |
| `--file`, `-f PATH` | Fitxategitik irakurri |
| `--stdin` | Stdin-etik irakurri |
| `--output`, `-o PATH` | Irteera fitxategi batean gorde |
| `--spell` | Ortografia zuzenketa gaitu (desgaituta lehenetsita) |
| `--no-punct` | Maiuskula/puntuazio zuzenketa desgaitu |
| `--quiet`, `-q` | Egoera mezuak isildu (stderr) |

### Adibide praktikoak

```bash
# Ahots-ezagutzaren irteera garbitu
uv run python txukun.py "euskal herrian euskaraz bizi nahi dugu"
# → Euskal Herrian euskaraz bizi nahi dugu.

# Ortografia bakarrik (eredua kargatu gabe)
uv run python txukun.py --no-punct --spell "akats bat dauka honek"
# → Akats bat dauka honek

# Fitxategi bat prozesatu emaitza gordez
uv run python txukun.py -f raw_text.txt -o clean_text.txt

# Hainbat fitxategi batera
for f in *.txt; do
  uv run python txukun.py -f "$f" -o "zuzendua/$f"
done
```

---

## Nola dabil?

1. **Cap+Punct eredua** (`itzune/txukun-cap-punct-eu`): `HiTZ/cap-punct-eu` MarianMT ereduaren bertsio kuantizatua (int8 ONNX, ~77 MB). Eredu originalak 9.78 milioi euskarazko esaldirekin entrenatu zuten HiTZ Zentroak (UPV/EHU). ONNX bertsio kuantizatua Itzune-k esportatu eta HF Hub-en argitaratu du.

2. **Ortografia zuzentzailea**: 160.000 hitzeko hiztegia (`data/eu-words.txt`), Xuxen hiztegitik eta ccmatrix corpusetik eraikia. Levenshtein distantzia ≤2 erabiltzen du iradokizunak aurkitzeko, corpus maiztasunaren arabera ordenatuta. Lehen iradokizuna automatikoki aplikatzen du `--spell` aktibatuta.

---

## Mugarik?

- **Lehen exekuzioan** ONNX eredua deskargatuko da (~77 MB) eta tokenizadorea HiTZ-en eredutik. Konexio ona behar da.
- **CPU soilik** exekutatzen da (`CPUExecutionProvider`)
- **Hallucination**: Ereduak hitz labur/arraroekin batzuetan hitz okerrak sortzen ditu (adib. "ausill-a", "IMAIO"). Jatorrizko PyTorch ereduaren berezko muga da.
- **ONNX int8 vs PyTorch**: Irteera ezberdina da PyTorch jatorrizkoarekiko. Esaldi ongi eratuekin ONNX bertsioak emaitza HOBEAK ematen ditu (adib. `"Euskal Herrian euskaraz bizi nahi dugu."` vs PyTorch-en `"EHE bizi nahi dugu."`)
- **Ortografia zuzenketa** ez da perfektua: deklinabide eta aditz-forma guztiak ez daude hiztegian. Akats nabarienak soilik zuzentzen ditu.

---

## Lizentzia

- **Kodea**: Apache 2.0
- **Eredu kuantizatua**: Apache 2.0 (`itzune/txukun-cap-punct-eu`)
- **Jatorrizko eredua**: Apache 2.0 (`HiTZ/cap-punct-eu`)
- **Hiztegia**: [Xuxen](https://github.com/itzune/dictionary-eu) (GPL)
