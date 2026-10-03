# readme_t3.md – Whisper-Aggregator T3 (T4-GPU, Timecodes, Sprecher-Erkennung)

Batch-Transkription für Podcast-MP3s mit **faster-whisper** auf der **Nvidia
Tesla T4**, inklusive optionalem **Timecode-** und **Sprecher-Output**
(PyAnnote-Diarization).

| Datei | Zweck |
|---|---|
| `whisper_aggregator_t3.py` | Das Skript (CLI, 1054 Zeilen) |
| `start.sh` | Fertiger Test-/ Produktionsaufruf mit sinnvollen Defaults |
| `requirements_t3.txt` | Python-Abhängigkeiten (getestete Versionen) |
| `readme_t3.md` | dieses Dokument |

---

## 1. Features

- **Rekursives Scan** eines Verzeichnisses nach `*.mp3` / `*.MP3`
- **GPU-first**: `--device auto` erkennt CUDA automatisch, nutzt `float16`
  auf der T4; `--device cuda` erzwingt die GPU, `--device cpu` die CPU
- **Je MP3 eine eigene Textdatei**, Verzeichnisstruktur 1:1 gespiegelt
- **Resume-fähig** + **atomares Schreiben** (`.part` → `rename`): Abbruch
  jederzeit möglich, ein erneuter Lauf setzt an der letzten Stelle fort
- **Aggregationsdatei** mit allen Transkripten (`--no-aggregate` abschaltbar)
- **Timecodes** pro Segment (`--timestamps`)
- **Sprecher-Erkennung** (`--enable-diarization`, PyAnnote) mit
  Overlap-Matching gegen die Whisper-Segmente
- **Fehlerresistent**: fehlgeschlagene Dateien werden protokolliert, der Lauf
  läuft weiter (Exit-Code 1 statt Abbruch)
- **Durchsatz-Messung**: Audio-Minuten, Rechenzeit, „x Echtzeit" im
  Fortschrittsbalken und in der Zusammenfassung
- **Zwei Container-Fallstricke werden automatisch entschärft** (siehe §9):
  fehlende `libcublas.so.12` und PyAV ≥ 19 ohne `metadata_errors`

---

## 2. Voraussetzungen

- Python ≥ 3.11 (getestet: 3.13)
- GPU empfohlen: Nvidia Tesla T4 oder besser, Treiber ≥ 580 für CUDA 13
- Netzwerkzugang beim ersten Lauf (Modell-Download von HuggingFace)
- Für Diarization zusätzlich: HuggingFace-Token + Freischaltung des Modells

---

## 3. Installation

```bash
pip install -r requirements_t3.txt
```

Prüfung:

```bash
python3 -c "import torch, faster_whisper; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
nvidia-smi          # muss die T4 zeigen
```

Ohne GPU funktioniert alles auch auf CPU, nur deutlich langsamer (siehe §8).

---

## 4. Schnellstart

```bash
cd /content
./start.sh                                        # voller Lauf, Plain
./start.sh --path "/content/PodcastBulkDownloader/mp3/CQ Morning Briefing"   # 1 Folge (~10 s)
DIARIZATION=1 ./start.sh                          # mit Sprecher-Erkennung
./start.sh --overwrite                            # alles neu erzeugen
```

`start.sh` zeigt vorab GPU, Zielpfade und den exakten Befehl, reicht eigene
Argumente an das Skript durch und meldet am Ende Exit-Code und Dauer.

**Umgebungsvariablen von `start.sh`:**

| Variable | Default | Bedeutung |
|---|---|---|
| `MP3_DIR` | `/content/PodcastBulkDownloader/mp3` | Eingabeverzeichnis |
| `TRANSCRIPT_DIR` | `transcript_t3` bzw. `transcript_t3_diar` | Zielordner (getrennt je Modus!) |
| `OUTPUT` | `ergebnis_t3.txt` bzw. `ergebnis_t3_diar.txt` | Aggregat |
| `MODEL` | `turbo` | Modellgröße |
| `DIARIZATION` | `auto` | `auto` = an, sobald `/content/.hf_token` existiert |
| `TOKEN_FILE` | `/content/.hf_token` | Token-Datei |
| `EXTRA_ARGS` | – | z. B. `"--beam-size 1 --language de"` |
| `PYTHON` | `python3` | anderer Interpreter |

Die Trennung der Zielordner ist Absicht: sonst würden nach einem Lauf **ohne**
Token Plain-Transkripte im Zielordner liegen und ein späterer Lauf **mit**
Token sie per Resume überspringen – ohne je Sprecher-Labels zu erzeugen.

---

## 5. Direktaufrufe (ohne `start.sh`)

```bash
# Plain, mit Auto-Gerätewahl
python3 whisper_aggregator_t3.py --path ./mp3 --transcript-dir ./transcript

# Timecodes
python3 whisper_aggregator_t3.py --path ./mp3 --timestamps

# Timecodes + Sprecher (Token-Datei)
python3 whisper_aggregator_t3.py --path ./mp3 \
  --enable-diarization --hf-token-file /content/.hf_token

# Nur Sprecher, ohne Timecodes
python3 whisper_aggregator_t3.py --path ./mp3 --enable-diarization --no-timestamps

# Erzwungene CPU, anderes Modell, Sprache fixieren
python3 whisper_aggregator_t3.py --path ./mp3 --device cpu --model-size base --language de
```

---

## 6. CLI-Referenz

| Flag | Default | Beschreibung |
|---|---|---|
| `--path` | *Pflicht* | Eingabeverzeichnis, rekursiv |
| `--transcript-dir` | `transcript` | Zielordner für Einzeltranskripte |
| `--output` | `ergebnis.txt` | Aggregat-Datei |
| `--no-aggregate` | aus | kein Aggregat schreiben |
| `--overwrite` | aus | Resume deaktivieren, alles neu erzeugen |
| `--model-size` | `turbo` | `tiny, base, small, medium, large, large-v3, large-v3-turbo, turbo` |
| `--device` | `auto` | `auto` (GPU wenn da) / `cuda` / `cpu` |
| `--device-index` | `0` | GPU-Index bei mehreren GPUs |
| `--beam-size` | `5` | `1` = schnellste auf GPU, `5` = beste Qualität |
| `--compute-type` | automatisch | `float16` (GPU), `int8` (CPU), jeder andere Wert per Override |
| `--language` | Auto-Erkennung | z. B. `de`, `en` |
| `--timestamps` / `--timecodes` | an bei Diarization | `[…] Text` pro Segment |
| `--no-timestamps` | – | Timecodes abschalten |
| `--enable-diarization` | aus | Sprecher-Erkennung (PyAnnote) |
| `--diarization-model` | `pyannote/speaker-diarization-3.1` | HF-Checkpoint |
| `--hf-token` | – | Token direkt (vorsicht: landet in der Shell-History) |
| `--hf-token-file` | – | Datei mit dem Token (empfohlen) |

---

## 7. Ausgabeformate

**Ohne Flags (Standard, unverändert zum Ursprungsskript):**
```text
Good morning. I'm Christina Karish and it's Wednesday, July 8th.
```

**`--timestamps`:**
```text
[0.00s - 5.02s] Good morning. I'm Christina Karish and it's Wednesday, July 8th.
[5.02s - 8.48s] morning briefing, and here are the top three headlines...
```

**`--enable-diarization` (setzt Timecodes automatisch an):**
```text
[0.00s - 5.02s] SPEAKER_00: Good morning. I'm Christina Karish...
[9.22s - 13.98s] SPEAKER_01: Supreme Court Justices Amy Coney Barrett...
```

Sprecher außerhalb der Diarization-Abdeckung werden als `UNKNOWN` markiert.

**Aggregationsdatei** (unverändertes Format):
```text
--- FILE: CQ Morning Briefing/Recess update....mp3 ---
<Transkript>
------------------------------------------
```

---

## 8. Performance (gemessen, Tesla T4)

| Szenario | Modell | Gerät | Ergebnis |
|---|---|---|---|
| 11 Folgen, 221,7 min Audio | turbo | T4 fp16 | **613 s = 21,7x Echtzeit**, 0 Fehler |
| Einzelfolge (1,9 min) | turbo | T4 fp16 | 4,8 s = **24,0x** |
| Einzelfolge mit Timecodes | turbo | T4 fp16 | 13,9x (Wort-Zeitstempel kosten was) |
| Einzelfolge (1,9 min) | turbo | CPU int8 | 195 s = **0,6x** |
| Einzelfolge (1,9 min) | tiny | CPU int8 | 19,9 s = 5,8x |

Auslastung im Lauf: ~69 % GPU, **1,9 GiB der 15 GiB VRAM** (Spielraum für
`large-v3`). Spracherkennung lief korrekt getrennt: `en (8x), de (3x)`.
Übersicht aller Blöcke im Aggregat: 11 Dateien, 37.023 Wörter, 217.510 Bytes.

> Faustregel für 3,7 h Podcast: **~10 min auf der T4** (ohne Diarization),
> statt ~6 h auf dieser CPU.

---

## 9. Diarization einrichten (einmalig)

**Schritt 1 – Modell freischalten** (gated, einmalig, braucht ein HF-Konto):
<https://huggingface.co/pyannote/speaker-diarization-3.1> → *Agree to access*.
Ebenso bei Bedarf `pyannote/segmentation-3.0`.

**Schritt 2 – Token besorgen.** Empfohlen ist die Token-Datei, weil ein Abruf
aus einem Nicht-Notebook-Prozess scheitert (`Secrets can only be fetched when
running from the Colab UI`). In einer **Zelle der Colab-UI** ausführen:

```python
from google.colab import userdata
import os
open('/content/.hf_token', 'w').write(userdata.get('HF_TOKEN'))
os.chmod('/content/.hf_token', 0o600)
print("geschrieben:", os.path.getsize('/content/.hf_token'), "Bytes")
```

**Schritt 3 – starten:**
```bash
DIARIZATION=1 ./start.sh
# oder automatisch (Default), sobald die Datei existiert:
./start.sh
```

Token-Reihenfolge im Skript: `--hf-token` → `--hf-token-file` →
Umgebungsvariable `HF_TOKEN` → Colab-Secret `HF_TOKEN`.

---

## 10. Resume, atomares Schreiben, Aggregat

- Ein Transkript gilt als fertig, sobald die Ziel-`.txt` existiert.
- Geschrieben wird `Ziel.txt.part` + `os.replace` → nie ein halbgeschriebenes
  File, ein Abbruch kann nie für „fertig" gehalten werden.
- Fehlgeschlagene Dateien erscheinen am Ende mit Pfad und Grund und werden
  beim nächsten Lauf automatisch erneut versucht (Exit-Code 1).
- Das Aggregat wird **immer** aus allen vorhandenen Einzeltranskripten neu
  gebaut, also auch dann, wenn in diesem Lauf nichts neu transkribiert wurde.

---

## 11. Exit-Codes

| Code | Bedeutung |
|---|---|
| 0 | Erfolg (auch „nichts zu tun") |
| 1 | mindestens eine Datei fehlgeschlagen (erneuter Lauf setzt fort) |
| 2 | `--path` existiert nicht oder ist kein Verzeichnis |
| 3 | Whisper-Modell nicht ladbar |
| 4 | Aggregat-Datei nicht schreibbar |
| 5 | `--device cuda` angegeben, aber keine GPU erkannt |
| 6 | Diarization-Konfiguration (kein Token, Token-Datei leer/unlesbar, Pipeline nicht ladbar) |

---

## 12. Troubleshooting

**`TypeError: open() got an unexpected keyword argument 'metadata_errors'`**
PyAV ≥ 19 hat dieses Argument entfernt, faster-whisper ruft es noch auf.
→ Wird vom Skript automatisch abgefangen (`ensure_pyav_compat()`: bei genau
diesem TypeError wird ohne das Argument wiederholt). Manuell prüfen: `av.__version__`.

**`Library libcublas.so.12 is not found or cannot be loaded`**
ctranslate2 ist ein CUDA-12-Build, die Bibliotheken liegen aber in
`<site-packages>/nvidia/*/lib` und fehlen im ldconfig-Cache.
→ Wird automatisch geladen (`ensure_cuda_runtime()` präloadet cudart, cublas,
cublasLt, cufft, curand, cusolver, cusparse, nvJitLink, cupti und hängt die
Pfade an `LD_LIBRARY_PATH`). Zeile im Log:
`CUDA-Laufzeit bereitgestellt: 9 NVIDIA-Bibliothek(en) geladen`.

**`GatedRepoError ... 401 ... Access to model ... is restricted` (Exit 6)**
Modell nicht freigeschaltet oder Token falsch/fehlend → §9, Schritt 1 + 2.

**`TimeoutException: Requesting secret HF_TOKEN timed out`**
Das Secret wurde aus einem Prozess abgefragt, der nicht die Colab-UI ist
(z. B. Shell oder eingeschobener Kernel-Aufruf). → §9, Schritt 2 (Token-Datei
über eine UI-Zelle erzeugen) oder `--hf-token`/`--hf-token-file` nutzen.

**`FEHLER: --device cuda ... keine CUDA-GPU erkannt` (Exit 5)**
`nvidia-smi` prüfen; `CUDA_VISIBLE_DEVICES` darf die GPU nicht ausblenden.

**pip-Warnung `numba 0.61.2 requires numpy<2.3, aber numpy 2.5.3`**
Kosmetisch: von `pyannote.audio` wurde numpy angehoben. Das Skript ist nicht
von numba abhängig; nur falls eine andere, externe Komponente numbra nutzt,
`pip install "numpy<2.3"` nachziehen.

**Torchaudio-Pyannote-Inkompatibilität**
torchaudio 2.11 kennt weder `AudioMetaData` noch `list_audio_backends`, was
pyannote erwartet. → `apply_torchaudio_patches()` setzt beides nach (Attribute,
die schon da sind, bleiben unangetastet).

**Erster Lauf lädt langsam**
Modelle werden von HuggingFace geladen (`turbo` ≈ 1,6 GiB). Der Cache liegt in
`~/.cache/huggingface/`, danach startet alles ohne Netz.

---

## 13. Code-Aufbau (wichtigste Stellen)

| Funktion | Aufgabe |
|---|---|
| `parse_args()` | CLI inkl. Timecode-/Diarization-Flags |
| `find_mp3_files()` / `plan_files()` | rekursives Suchen + Resume-Planung |
| `write_text_atomic()` | `.part` → `rename` |
| `resolve_device()` / `describe_gpu()` | Geräteauto-Erkennung, GPU-Header |
| `ensure_cuda_runtime()` | CUDA-12-Libraries präladen |
| `ensure_pyav_compat()` | PyAV-≥19-Brücke für `metadata_errors` |
| `load_model()` | Whisper-Modell (GPU/CPU) |
| `get_hf_token()` | Token-Reihenfolge CLI → Datei → Env → Colab-Secret |
| `apply_torchaudio_patches()` | torchaudio/pyannote-Kompatibilität |
| `load_diarization_pipeline()` | PyAnnote laden + auf GPU legen |
| `get_speaker_for_time()` | Overlap-Matching Segment ↔ Sprecher |
| `run_diarization()` | Sprechertrennung mit Fehler-Weiterlaufen |
| `transcribe_file()` | A) Diarization → B) Transkription → C) Formatierung |
| `build_aggregate()` | Aggregat aus allen Einzeltranskripten |
| `main()` | Workflow, Statistik, Exit-Codes |

---

## 14. Änderungen gegenüber dem Ursprungsskript

1. **GPU-Umstellung**: `--device auto` (T4 + `float16`), `--device-index`,
   `--beam-size`, Modellwahl um `large-v3`/`large-v3-turbo`, Default `turbo`.
2. **Zwei Container-Fallstricke entschärft**: `libcublas.so.12`-Preload und
   PyAV-`metadata_errors`-Brücke.
3. **Zeitmessung**: Durchsatz in Fortschritt und Zusammenfassung.
4. **Bugfix**: tqdm-Fallback referenzierte die nicht existierende Variable
   `files` (jetzt `items`).
5. **Neu**: Timecodes, Diarization, Token-Reihenfolge, `start.sh`.
