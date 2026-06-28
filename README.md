# Txukun CLI

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://docs.astral.sh/uv/)

Euskarazko testuen **maiuskulak, puntuazioa eta ortografia** zuzentzeko komando-lerroko tresna.

- 🧠 **Cap+Punct**: `itzune/txukun-cap-punct-eu` ONNX int8 eredu kuantizatua (~77 MB, `optimum[onnxruntime]`)
- 🔍 **Ortografia**: Hunspell + Xuxen euskal hiztegia (142k sarrera + eratorpen-arau afixuak)
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

2. **Ortografia zuzentzailea**: Hunspell bidezko zuzentzaile ortografikoa, [Xuxen](https://xuxen.eus/) euskal hiztegiarekin (`data/eu.aff` + `data/eu.dic`, 142k sarrera). Hiztegi afixu-arauen bidez euskal morfologia aberatsa kudeatzen du (deklinabideak, aditz-formak, hitz-elkarketak).

---

## ⚠️ Mugak eta Oharrak

### 🔴 Aluzinazioak

[`HiTZ/cap-punct-eu`](https://huggingface.co/HiTZ/cap-punct-eu) ereduak **aluzinazioak** sor ditzake — existitzen ez diren hitzak asmatzea — bereziki testu labur, arraro edo ohiz kanpokoa denean. AI eredu sortzaile guztien berezko arazoa da. ONNX int8 kuantizazioak aluzinazio horien forma alda dezake (beste txorakeria batzuk) baina ez du jatorrizko arazoa konpontzen.

Emaitza onenak lortzeko, esaldi oso eta ongi eratuak erabili.

### 🟡 Ortografia zuzentzailea: Hunspell + Xuxen

Txukun-en ortografia zuzentzaileak **[Hunspell](https://hunspell.github.io/)** erabiltzen du — ortografia zuzentzaileen estandar irekia — **[Xuxen](https://xuxen.eus/)** euskal hiztegiarekin. Xuxen Elhuyar-ek eta UPV/EHUko IXA taldeak garatzen dute.

Hiztegi afixu-arauen bidez euskal morfologia aberatsa kudeatzen du:
- Deklinabide guztiak: `etxea`, `etxearekin`, `etxeetara`...
- Hitz-elkarketak: `hitz-armak`, `etxe-aurrean`...
- Aditz-formak: `zetozen`, `genbiltzan`...

**Ez da AI edo LLM**: arau linguistikoetan oinarritzen da, ez machine learning-ean.

### 🟦 Eremua

Txukun **euskarazko testuetarako** (`eu`/`eus`) diseinatuta dago. Ez du beste hizkuntzekin behar bezala funtzionatuko.

### 📦 ONNX int8 vs PyTorch jatorrizkoa

ONNX int8 bertsio kuantizatuak irteera ezberdina ematen du PyTorch jatorrizkoarekin alderatuta. Esaldi ongi eratuekin ONNX bertsioak emaitza **hobeak** ematen ditu (adib. `"Euskal Herrian euskaraz bizi nahi dugu."` vs PyTorch-en `"EHE bizi nahi dugu."`).

---

## Lizentzia

- **Kodea**: Apache 2.0
- **Eredu kuantizatua**: Apache 2.0 (`itzune/txukun-cap-punct-eu`)
- **Jatorrizko eredua**: Apache 2.0 (`HiTZ/cap-punct-eu`)
- **Hiztegia**: [Xuxen](https://github.com/itzune/dictionary-eu) (GPL)
