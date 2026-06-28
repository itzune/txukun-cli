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

### 1. Klonatu eta instalatu Python mendekotasunak

```bash
git clone https://github.com/itzune/txukun-cli.git
cd txukun-cli
uv sync
```

`uv sync` komandoak beharrezko Python mendekotasunak instalatuko ditu:
- `optimum[onnxruntime]` — ONNX eredua exekutatzeko (CPU, ~77 MB deskarga)
- `click` — CLI interfazerako

### 2. Instalatu Hunspell (ortografia zuzenketarako)

Ortografia zuzenketa (`--spell`) erabiltzeko **Hunspell** behar da sisteman:

```bash
# Debian / Ubuntu
sudo apt install hunspell

# macOS
brew install hunspell

# Arch Linux
sudo pacman -S hunspell
```

> **Oharra**: Hunspell gabe ere, `txukun.py`-k cap+punct zuzenketa egiten du normaltasunez. `--spell` aukera erabiltzean Hunspell ez badago eskuragarri, abisu bat erakutsiko du eta ortografia zuzenketarik gabe jarraituko du.

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

### Fluxua

```
--spell GABE:  [sarrera] → cap+punct ONNX → [irteera]
--spell EZARRI: [sarrera] → Hunspell zuzenketa → cap+punct ONNX → [irteera]
```

### 1. Cap+Punct eredua

`itzune/txukun-cap-punct-eu` — `HiTZ/cap-punct-eu` MarianMT ereduaren bertsio kuantizatua (int8 ONNX, ~77 MB). Eredu originalak 9.78 milioi euskarazko esaldirekin entrenatu zuten HiTZ Zentroak (UPV/EHU). ONNX bertsio kuantizatua Itzune-k esportatu eta HF Hub-en argitaratu du.

Ereduak sarrerako testu gordina (minuskuletan, puntuaziorik gabe) hartu eta maiuskula eta puntuazio egokiak gehitzen dizkio:

```
"euskal herrian euskaraz bizi nahi dugu" → "Euskal Herrian euskaraz bizi nahi dugu."
```

### 2. Ortografia zuzentzailea

`--spell` ezartzean, **lehenik ortografia zuzentzen da**, eta gero testu zuzendua cap+punct ereduari pasatzen zaio:

```bash
uv run python txukun.py --spell "ser gertatu da hemen"
# Ortografia:  ser → zer
# Cap+punct:   "zer gertatu da hemen" → "Zer gertatu da hemen?"
```

Fluxu honek ereduak sarrera garbiagoa jasotzea ahalbidetzen du, aluzinazio-arriskua murriztuz (ortografia-akatsak dituzten hitzek maiz aluzinazioak eragiten baitituzte).

Zuzentzaileak **[Hunspell](https://hunspell.github.io/)** erabiltzen du — estandar irekia mundu osoko hizkuntzetarako — **[Xuxen](https://xuxen.eus/)** euskal hiztegiarekin. Xuxen Elhuyar-ek eta UPV/EHUko IXA taldeak garatzen dute.

Hiztegi-fitxategiak (`data/eu.aff` + `data/eu.dic`, 142k sarrera + eratorpen-arauak) paketean bertan datoz. Hunspell `-a` (ispell pipe modua) bidez persistente exekutatzen da backend prozesu gisa, hitz bakoitzeko ~0.1ms latentzia lortuz.

Hiztegiaren afixu-arauen bidez euskal morfologia aberatsa kudeatzen du, hitz guztiak banan-banan zerrendatu beharrik gabe:

- **Deklinabideak**: `etxe`, `etxea`, `etxearekin`, `etxeetara`, `etxeetan`... (erro batetik milaka forma)
- **Aditz-formak**: `zetozen`, `genbiltzan`, `dizkizuegu`...
- **Hitz-elkarketak**: `hitz-armak`, `etxe-aurrean`, `sare-arloa`...

**Ez da AI edo LLM**: Hunspell arau linguistikoetan oinarritutako motor determinista bat da. Ez du machine learning-ik, entrenamendurik edo eredu estatistikorik erabiltzen — hiztegi bat (142k hitz) eta euskal morfologia deskribatzen duten afixu-arauak (`data/eu.aff`) baino ez. Akats bat aurkitzean, Hunspell-en iradokizun-motorrak editatzeko distantzia erabiltzen du antzeko hitz zuzenak aurkitzeko, eta lehen iradokizuna automatikoki aplikatzen da.

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

**Ez da AI edo LLM**: Hunspell arau linguistikoetan eta hiztegi batean oinarritutako motor determinista da. Ez du machine learning-ik, entrenamendurik edo eredu estatistikorik erabiltzen.

### 🟦 Eremua

Txukun **euskarazko testuetarako** (`eu`/`eus`) diseinatuta dago. Ez du beste hizkuntzekin behar bezala funtzionatuko.

### 📦 ONNX int8 vs PyTorch jatorrizkoa

ONNX int8 bertsio kuantizatuaren eta PyTorch jatorrizkoaren arteko irteerak ezberdinak dira. Ebaluazio formalik EZ da egin, baina gure probetan esaldi ongi eratuekin ONNX bertsioak emaitza hobeak ematen dituela dirudi (adib. `"Euskal Herrian euskaraz bizi nahi dugu."` vs PyTorch-en `"EHE bizi nahi dugu."`).

---

## Lizentzia

- **Kodea**: Apache 2.0
- **Eredu kuantizatua**: Apache 2.0 (`itzune/txukun-cap-punct-eu`)
- **Jatorrizko eredua**: Apache 2.0 (`HiTZ/cap-punct-eu`)
- **Hiztegia**: [Xuxen](https://github.com/itzune/dictionary-eu) (GPL)
