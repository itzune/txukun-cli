# Txukun CLI

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://docs.astral.sh/uv/)

Euskarazko testuen **maiuskulak, puntuazioa, ortografia eta gramatika** zuzentzeko komando-lerroko tresna.

Web bertsioaren ([txukun](https://github.com/itzune/txukun)) hiru eredu berberak erabiltzen ditu, ONNX Runtime bidez CPU-n exekutatzen direnak:

- 🧠 **Cap+Punct** — maiuskulak eta puntuazioa (MarianMT ONNX int8, ~77 MB)
- 🔍 **Ortografia** — Hunspell + maiztasun berriro-ordenaketa + BERTeus ONNX berriro-ordenaketa neuronala (int4, 85 MB + 74 MB embeddings)
- ✍️ **Gramatika** — GECToR akats gramatikalak zuzentzeko (RoBERTa-eus-base, int4 ONNX, ~85 MB)

> **Itzune** kolektiboaren proiektua — euskarako AI tresna libre eta pribatuak.

---

## 🧠 Hiru eredu, hiru geruza

| # | Eredua | Rola | Tamaina |
|---|---|---|---|
| 1 | **[cap-punct-eu](https://huggingface.co/itzune/txukun-cap-punct-eu)** | Maiuskulak eta puntuazioa | ~77 MB (int8) |
| 2 | **[berteus-onnx](https://huggingface.co/itzune/berteus-onnx)** | Ortografia: hautagaien berriro-ordenaketa neuronala | 85 MB (int4) + 74 MB |
| 3 | **[gector-eus-onnx](https://huggingface.co/itzune/gector-eus-onnx)** | Gramatika zuzenketa + akats-detekzioa | ~85 MB (int4) |

Hiru ereduak **HF Hub-etik automatikoki deskargatzen dira** lehen exekuzioan, eta cache-atseginak dira hurrengo exekuzioetarako.

---

## Instalazioa

### 1. Klonatu eta instalatu Python mendekotasunak

```bash
git clone https://github.com/itzune/txukun-cli.git
cd txukun-cli
uv sync
```

### 2. Instalatu Hunspell (ortografia zuzenketarako)

```bash
# Debian / Ubuntu
sudo apt install hunspell

# macOS
brew install hunspell

# Arch Linux
sudo pacman -S hunspell
```

> Hunspell gabe ere, cap-punct eta gramatika zuzenketa funtzionatzen dute. Ortografia zuzenketa desgaituta geratzen da.

---

## Erabilera

### Zuzenketa (lehenetsia)

```bash
uv run python txukun.py "gaur goizean ama egin du bazkaria eta gero etsea garbitu dut"
# → Gaur goizean amak egin du bazkaria eta gero etxea garbitu dut.
```

### Detekzioa (JSON irteera)

```bash
uv run python txukun.py -d "gaur goizean ama egin du bazkaria"
```

```json
[
  { "id": "e1", "frm": 0, "to": 4, "original": "gaur", "suggestion": "Gaur",
    "category": "cappunct", "title": "Maiuskula", "context": "" },
  { "id": "e2", "frm": 13, "to": 16, "original": "ama", "suggestion": "amak",
    "category": "grammar", "title": "Gramatika", "context": "gaur goizean" }
]
```

### Ereduak aukeratu

```bash
# Eredu jakin batzuk bakarrik
uv run python txukun.py --enable spell --enable cappunct "text"

# Eredu bat kendu
uv run python txukun.py --disable grammar "text"
```

### Fitxategiak eta stdin

```bash
uv run python txukun.py -f input.txt -o output.txt
cat testua.txt | uv run python txukun.py --stdin
echo "kaixo" | uv run python txukun.py --stdin -q
```

### Aukerak

| Aukera | Deskribapena |
|---|---|
| `TEXT` | Zuzendu beharreko testua |
| `-d`, `--detect` | JSON irteera akatsekin (posizioa, iradokizuna, kategoria) |
| `-c`, `--correct` | Testu zuzendua irteera (lehenetsia) |
| `--enable MODEL` | Eredu jakin batzuk bakarrik gaitu (`cap-punct`, `spell`, `grammar`) |
| `--disable MODEL` | Eredu bat desgaitu |
| `-f`, `--file PATH` | Fitxategitik irakurri |
| `--stdin` | Stdin-etik irakurri |
| `-o`, `--output PATH` | Irteera fitxategian gorde |
| `-q`, `--quiet` | Egoera mezuak isildu |

---

## Nola dabil?

### Fluxua

```
[sarrera] → strip_markdown → 3 detektore (sekuentzialki) → merge → [irteera]
                                │
                                ├─ GECToR: gramatika zuzenketa → diffWords → replace changes
                                ├─ Hunspell: akats ortografikoak → Tier1 (freq+ed) + Tier2 (BERTeus)
                                └─ MarianMT: cap-punct → diffWords → case/punct-only changes
```

Hiru detektoreek **testu arruntan** (markdown syntaxia kenduta) egiten dute lan, eta akatsen posizioak jatorrizko testura mapatzen dira (`strip_markdown`-en posizio-maparen bidez).

### Markdown euskera

Testuak markdown sintaxia badu (`#`, `**`, `[]`, etab.), automatikoki garbitzen da ereduei pasatu aurretik. Akatsen posizioak jatorrizko markdown-era mapatzen dira:

```bash
uv run python txukun.py -d "# Nire txostena

laister iritsiko naiz"
# → laister → Laster (spelling+cap-punct mergea, posizioa jatorrizko markdown-en)
```

### Akatsen mergea

Akats bat posizio berean hainbat detektorek aurkitzen badute, mergeatu egiten dira:
- **Ortografia + Cap-punct**: `laister` → `Laster` (spell: `laster` + cap: `Laister`)
- **Gramatika + Cap-punct**: `ama` → `Amak` (grammar: `amak` + cap: `Ama`)

### Heading puntuazioa

Izenburuetan (`#`), puntuazio-iradokizunak ezabatzen dira (ez da puntua gehitzen izenburuaren bukaeran).

---

## ⚠️ Mugak

### Aluzinazioak

Cap-punct (MarianMT) eta GECToR ereduek aluzinazioak sor ditzakete. `constrain_lcs()` funtzioak (LCS lerrokatzea) cap-punct-en hitz-ordezkapenak arbuiatzen ditu — maiuskula/puntuazio aldaketak bakarrik onartzen ditu.

### Eremua

Euskarazko testuetarako diseinatuta. Ez du beste hizkuntzekin funtzionatuko.

### Errealword-ak (real-word errors)

GECToR-eusek ezin ditu hitz errealen akatsak detektatu (adib. `hura` vs `ura`), Elhuyar entrenamendu-datuetan ez baitaude akats mota hori. Ikusi [gector-eus/TODO.md](https://github.com/itzune/gector-eus/blob/main/TODO.md).

---

## 🛡️ Akats-kudeaketa eta fallback-a

Eredu bakoitza **lazy-loading** da eta hutsegiteetan **graceful degradation** aplikatzen da:

| Osagaia | Hutsegitea | Fallback |
|---|---|---|
| Hunspell ez instalatuta | `hunspell` komandoa ez da aurkitzen | Ortografia desgaituta, cap-punct + gramatika jarraitzen dute |
| BERTeus kargatzeak huts | ONNX/deskarga errorea | Tier 1 (freq+ed) bakarrik erabiltzen da |
| GECToR kargatzeak huts | ONNX/deskarga errorea | Gramatika desgaituta, besteak jarraitzen dute |
| Cap-punct kargatzeak huts | ONNX/deskarga errorea | Cap-punct desgaituta, besteak jarraitzen dute |

---

## 🎯 Konfiantza-iragazkia (confidence filtering)

Eredu bakoitzak konfiantza-puntuazio bat ematen du iradokizun bakoitzeko. Balio baxuko iradokizunak automatikoki ezabatzen dira, **over-correction** eta **false positive** arazoak murrizteko.

| Eredua | Konfiantza-iturria | Atalasea |
|---|---|---|
| GECToR (gramatika) | P(INCORRECT) detekzio-burutik (0.0–1.0) | **0.05** |
| BERTeus (ortografia) | Kosinu antzekotasuna, normalizatuta (0–1) | **0.50** |
| MarianMT (cap-punct) | LCS lerrokatze-tasa (1.0 = hitz-ordezkapenik ez) | **1.00** |

Atalase hauek 220 kasuko ebaluazio-datu-sortan kalibratu dira (`tests/gec-benchmark/eval_dataset.json`), grid search bidez (`tests/gec-benchmark/confidence_per_model.py`).

**Emaitza**: 22.7% → 38.6% zehaztasuna (+15.9 puntu), over-correction 139→66 eta false positive 12→1 murriztuz.

> ⚠️ **Ereduak eguneratzen badira**, atalase hauek berrikustekoak dira. Berriro exekutatu:
> ```bash
> uv run python tests/gec-benchmark/run_eval.py --output /tmp/eval_results.json
> uv run python tests/gec-benchmark/confidence_per_model.py
> ```

Config-ak `txukun_lib/analyze.py`-n daude (`CONFIDENCE_THRESHOLDS` aldagaia).

---

## Lizentzia

- **Kodea**: Apache 2.0
- **cap-punct eredua**: Apache 2.0 (`itzune/txukun-cap-punct-eu`, jatorrizkoa `HiTZ/cap-punct-eu`)
- **BERTeus eredua**: CC-BY-NC-SA 4.0 (`itzune/berteus-onnx`, jatorrizkoa `ixa-ehu/berteus-base-cased`)
- **GECToR eredua**: CC-BY-NC-SA 4.0 (`itzune/gector-eus-onnx`, Elhuyar GEC datuekin entrenatua)
- **Hiztegia**: [Xuxen](https://github.com/itzune/dictionary-eu) (GPL)

---

## 🔗 Erlazionatutako proiektuak

- **[txukun](https://github.com/itzune/txukun)** — Web bertsioa (3 eredu, Grammarly-style UI)
- **[gector-eus](https://github.com/itzune/gector-eus)** — GECToR Basque entrenamendua
- [itzune/berteus-onnx](https://huggingface.co/itzune/berteus-onnx) — BERTeus int4 ONNX
- [itzune/gector-eus-onnx](https://huggingface.co/itzune/gector-eus-onnx) — GECToR int4 ONNX
- [itzune/txukun-cap-punct-eu](https://huggingface.co/itzune/txukun-cap-punct-eu) — Cap-punct int8 ONNX
- [Parakeet-eu](https://github.com/itzune/parakeet-eu) — Euskarazko ASR
