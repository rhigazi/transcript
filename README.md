# Whisper-Aggregator-CLI

**Projektname:** `Whisper-Aggregator-CLI`
**Sprache:** Python 3.x · **Interface:** Command Line Interface (CLI)

Kommandozeilen-Tool, das eine ganze Verzeichnis-Hierarchie nach MP3-Dateien
durchsucht, jede Datei mit einem Whisper-Modell
(**[faster-whisper](https://github.com/SYSTRAN/faster-whisper)**) transkribiert
und die Ergebnisse zweifach ablegt:

1. **pro MP3 eine eigene Textdatei** im Transcript-Verzeichnis
   (Verzeichnisstruktur 1:1 gespiegelt) → **resume-fähig**
2. optional **eine aggregierte Gesamtdatei** (`ergebnis.txt`)

---

## Features

- 🔍 **Rekursiver Datei-Scan** über `pathlib` – findet MP3s auch in
  Unterverzeichnissen (`.mp3` und `.MP3`)
- 🧠 **Transkriptions-Engine** mit `faster-whisper` (CTranslate2-basiert,
  deutlich schneller als die Original-Whisper-Implementierung)
- 📁 **Einzeltranskripte** in `transcript/` – Verzeichnisbaum des Inputs
  wird 1:1 gespiegelt, Suffix `.mp3` → `.txt`
- ▶️ **Resume / Fortsetzen:** Vorhandene Transkripte werden übersprungen.
  Bricht der Lauf ab (Absturz, Stromausfall, `Ctrl+C`), wird beim nächsten
  Start an der letzten Stelle weitergemacht – nichts wird neu transkribiert
- 💾 **Atomares Schreiben** (`.part`-Datei + rename): ein abgebrochener Lauf
  hinterlässt nie ein halbgeschriebenes Transkript, das für „fertig" gehalten
  würde
- 📦 **Aggregation** aller Transkripte in eine globale Textdatei
  (`ergebnis.txt`, auch nach einem Resume korrekt vollständig)
- 🛡️ **Fehlerresilienz** – eine korrupte Datei stoppt den Lauf nicht, der
  Fehler wird geloggt, die nächste Datei wird verarbeitet; fehlgeschlagene
  Dateien haben kein Transkript und werden beim nächsten Lauf erneut versucht
- 📊 **Fortschrittsbalken** via `tqdm` inkl. laufender Erfolgs-/Fehler-/
  Übersprungen-Zählung
- ⚙️ **Konfigurierbar** – Modellgröße, Gerät (CPU/GPU), Rechentyp, Sprache

---

## Voraussetzungen

- Python **3.9+**
- `pip`
- Optional für GPU-Betrieb: eine CUDA-fähige NVIDIA-GPU mit passenden
  [CUDA-Laufzeitbibliotheken von NVIDIA](https://nvidia.github.io/cuda-wheels/)
  (`ctranslate2` erkennt diese automatisch)

## Installation

```bash
git clone <repo-url>            # optional
cd Whisper-Aggregator-CLI
pip install -r requirements.txt
```

`requirements.txt`:

```text
faster-whisper>=1.0.0
av>=12,<19
tqdm>=4.66.0
```

> **Wichtig:** `av` ist bewusst auf `<19` gepinnt. Ab `av` 19.x wurde das
> von `faster-whisper` genutzte `metadata_errors`-Argument aus `av.open()`
> entfernt – jede Transkription scheitert sonst mit
> `TypeError: open() got an unexpected keyword argument 'metadata_errors'`.

Beim ersten Lauf lädt das gewählte Whisper-Modell automatisch von der
Hugging Face Hub (z. B. `tiny` ≈ 75 MB, `large-v3` ≈ 3 GB). Meldungen wie
`Warning: You are sending unauthenticated requests to the HF Hub` sind
harmlos; mit einem `HF_TOKEN` werden sie abgestellt.

---

## Schnellstart mit `start.sh`

Das mitgelieferte Starter-Skript prüft die Umgebung, zeigt den Resume-Stand
und startet den Lauf (inkl. Protokoll nach `whisper_run.log`):

```bash
./start.sh                     # Standard: PodcastBulkDownloader/mp3 -> transcript/
./start.sh --overwrite         # alles neu transkribieren
./start.sh --model-size base   # andere Modellgröße
./start.sh --help              # alle Optionen des Python-Tools
```

Konfiguration über Umgebungsvariablen (Vorfallen überschreiben):

```bash
INPUT_PATH=/pfad/zu/mp3s MODEL_SIZE=small DEVICE=cuda ./start.sh
INPUT_PATH=... WHISPER_LANGUAGE=de ./start.sh     # Sprache erzwingen (sonst Auto)
```

| Variable           | Default                    | Bedeutung                                   |
|--------------------|----------------------------|---------------------------------------------|
| `INPUT_PATH`       | `PodcastBulkDownloader/mp3` | Root-Verzeichnis der MP3s                  |
| `TRANSCRIPT_DIR`   | `transcript`               | Zielordner der Einzeltranskripte            |
| `OUTPUT`           | `ergebnis.txt`             | Aggregat-Datei                              |
| `MODEL_SIZE`       | `tiny`                     | `tiny`\|`base`\|`small`\|`medium`\|`large`\|`turbo` |
| `DEVICE`           | `cpu`                      | `cpu` oder `cuda`                           |
| `WHISPER_LANGUAGE` | leer (Auto)                | Sprachcode, z. B. `de` – `de_DE` wird zu `de` normalisiert |
| `LOGFILE`          | `whisper_run.log`          | Lauf-Protokoll (wird je Lauf überschrieben) |

> `LANGUAGE` wird bewusst **nicht** als Variable verwendet – das ist die
> System-Locale und würde als Whisper-Sprachcode fehlschlagen.

**Abbruch/Weiterführung:** `Ctrl+C` oder ein Absturz ist unkritisch – einfach
`./start.sh` erneut ausführen, bereits vorhandene Transkripte werden
übersprungen (die Startanzeige zeigt vorab, wie viele Dateien noch fehlen).

---

## Verwendung (direkt per Python)

```bash
python whisper_aggregator.py --path ./podcasts
```

### Beispiel: kompletter Lauf

```bash
python whisper_aggregator.py \
  --path ./podcasts \
  --transcript-dir ./transcript \
  --output ./ergebnis.txt \
  --model-size tiny \
  --device cpu \
  --language de
```

### Nach einem Abbruch fortsetzen

Einfach denselben Befehl wiederholen – bereits vorhandene Transkripte
werden übersprungen:

```bash
# Start
python whisper_aggregator.py --path ./podcasts --model-size tiny
# ... Lauf bricht ab ...
# Fortsetzung: transkribiert nur noch die fehlenden Dateien
python whisper_aggregator.py --path ./podcasts --model-size tiny
```

### Alle Parameter

| Parameter          | Pflicht | Default        | Beschreibung                                                       |
|--------------------|---------|----------------|--------------------------------------------------------------------|
| `--path`           | ja      | –              | Root-Verzeichnis (Input), wird rekursiv durchsucht                 |
| `--transcript-dir` | nein    | `transcript`   | Verzeichnis für die EINZELNEN Transkripte (Struktur 1:1 gespiegelt) |
| `--output`         | nein    | `ergebnis.txt` | Aggregierte Ziel-Textdatei (Ordner werden automatisch angelegt)    |
| `--no-aggregate`   | nein    | aus            | Keine aggregierte Datei schreiben, nur Einzeltranskripte           |
| `--overwrite`      | nein    | aus            | Vorhandene Transkripte neu erzeugen (Resume deaktivieren)          |
| `--model-size`     | nein    | `base`         | `tiny`, `base`, `small`, `medium`, `large`, `turbo`                |
| `--device`         | nein    | `cpu`          | `cpu` oder `cuda` (GPU)                                            |
| `--compute-type`   | nein    | automatisch    | `int8` (CPU) bzw. `float16` (CUDA), manuell überschreibbar         |
| `--language`       | nein    | Auto-Erkennung | Sprachcode, z. B. `de`, `en`                                       |
| `-h`, `--help`     | –       | –              | Alle Optionen anzeigen                                             |

### Modellgrößen im Überblick

| Modell   | Größe ca. | Geschwindigkeit | Genauigkeit | Empfehlung                          |
|----------|-----------|-----------------|-------------|-------------------------------------|
| `tiny`   | 75 MB     | sehr schnell    | grob        | Tests, Skript-Checks                |
| `base`   | 145 MB    | schnell         | gut         | **Default**, schnelle Vorab-Läufe   |
| `small`  | 484 MB    | mittel          | sehr gut    | gute Allround-Wahl (CPU)            |
| `medium` | 1.5 GB    | langsam         | hoch        | schwierige Aufnahmen (CPU/GPU)      |
| `large`  | 3 GB      | sehr langsam    | am höchsten | beste Qualität (GPU empfohlen)      |
| `turbo`  | 1.6 GB    | ~8× schneller   | fast wie large | **Empfohlen bei GPU** („Turbo"-Effekt) |

### Beispielausgabe in der Konsole

```text
Gefunden: 11 MP3-Datei(en) unter /content/PodcastBulkDownloader/mp3
Resume: 4 Transkript(e) vorhanden in 'transcript' -> werden übersprungen (mit --overwrite neu erzeugen)
Lade Whisper-Modell 'tiny' auf CPU (compute_type=int8) ...
Transkribiere: 100%|██████████████| 7/7 [28:11<00:00, ok=7 Fehler=0 überspr.4]

==================================================
ZUSAMMENFASSUNG
==================================================
  Verzeichnis      : /content/PodcastBulkDownloader/mp3
  Modell           : tiny @ cpu (int8)
  Dateien gesamt   : 11
  Neu transkribiert: 7
  Übersprungen     : 4
  Fehlgeschlagen   : 0
  Transkript-Ordner: /content/transcript (11 Datei(en))
  Aggregat-Datei   : /content/ergebnis.txt
```

---

## Ausgabe

### 1. Einzeltranskripte (immer)

Verzeichnisstruktur des Inputs 1:1, nur das Suffix ändert sich:

```text
Eingabe/                                    Transcript/
├── Podcast A/                              ├── Podcast A/
│   ├── folge-01.mp3                        │   └── folge-01.txt
│   └── folge-02.mp3            ──►         ├── Podcast A/
├── Podcast B/                              │   └── folge-02.txt
│   └── special.mp3                         └── Podcast B/
└── root-episode.mp3                            └── special.txt
                                                 root-episode.txt
```

Jede `.txt` enthält ausschließlich den Transkript-Text dieser MP3.

### 2. Aggregat-Datei (`--output`, Standard `ergebnis.txt`)

```text
--- FILE: [Pfad relativ zu --path] ---
[Transkript-Inhalt]
------------------------------------------
```

Für Dateien direkt im Root entspricht das dem Dateinamen, für Dateien in
Unterverzeichnissen steht der relative Pfad darin (so bleiben doppelte
Dateinamen unterscheidbar):

```text
--- FILE: root-episode.mp3 ---
...
--- FILE: Podcast A/folge-01.mp3 ---
...
```

---

## Workflow (Logik-Schema)

```text
1. CLI-Argumente parsen (Pfad, Transcript-Dir, Output, Modell, Gerät …)
2. Alle *.mp3-Dateien via pathlib.glob("**/*.mp3") finden
3. Resume-Planung: vorhandene .txt -> skip, fehlende -> todo
4. Whisper-Modell initialisieren (CPU/GPU), falls todo != leer
5. Für jede zu verarbeitende Datei:
   a. Transkription durchführen -> text_content
   b. Atomar als eigene .txt ins Transcript-Verzeichnis schreiben
   c. Bei Fehler: loggen, keine .txt anlegen, zur nächsten Datei springen
6. Alle vorhandenen Einzeltranskripte in 'ergebnis.txt' aggregieren
   (überspringen mit --no-aggregate)
7. Zusammenfassung (neu / übersprungen / fehlgeschlagen) ausgeben
```

## Exit-Codes

| Code | Bedeutung                                                          |
|------|--------------------------------------------------------------------|
| `0`  | Erfolgreich – alle Dateien transkribiert (bzw. schon vorhanden)     |
| `1`  | Mindestens eine Datei konnte nicht transkribiert werden             |
| `2`  | Ungültiger Pfad (nicht vorhanden oder kein Verzeichnis)             |
| `3`  | Whisper-Modell konnte nicht geladen werden                          |
| `4`  | Aggregat-Datei konnte nicht geschrieben werden                      |

Damit eignet sich das Tool direkt für Shell-Pipelines:

```bash
python whisper_aggregator.py --path ./podcasts || echo "Teilweise Fehler!"
```

## Projektstruktur

```text
Whisper-Aggregator-CLI/
├── start.sh                 # Starter (Checks, Resume-Anzeige, Log)
├── whisper_aggregator.py   # CLI-Tool (Argumente, Scan, Transkription, Output)
├── requirements.txt        # Abhängigkeiten (faster-whisper, av<19, tqdm)
├── README.md               # Diese Dokumentation
├── prd.md                  # Produktanforderungsdokument
├── transcript/             # Einzeltranskripte (wird beim Lauf erzeugt)
├── ergebnis.txt            # Aggregat (wird beim Lauf erzeugt)
└── whisper_run.log         # Lauf-Protokoll (wird beim Lauf erzeugt)
```

## Troubleshooting

| Problem | Ursache / Lösung |
|---------|------------------|
| `ModuleNotFoundError: No module named 'faster_whisper'` | `pip install -r requirements.txt` ausführen |
| `TypeError: open() got an unexpected keyword argument 'metadata_errors'` | `pip install "av<19"` – av 19.x ist mit faster-whisper inkompatibel |
| `FEHLER: Eingabeverzeichnis nicht gefunden` | Pfad in `--path` prüfen (muss ein Verzeichnis sein) |
| `Keine MP3-Dateien gefunden` | Verzeichnis enthält keine `.mp3` – Pfad oder Dateitypen prüfen |
| Lauf startet „auf einmal" schnell durch | Resume greift – alle Transkripte sind schon vorhanden (gewünschtes Verhalten) |
| Transkript soll neu erzeugt werden | `--overwrite` hinzufügen |
| CUDA-Fehler beim Modell-Lauf | `--device cpu` verwenden oder CUDA-Laufzeit/`ctranslate2`-Version prüfen |
| `[FEHLER] ... nicht transkribieren` | Datei ist beschädigt/nicht lesbar – Lauf läuft trotzdem weiter, beim nächsten Start wird sie erneut versucht |
| Verwaiste `*.txt.part`-Dateien | Können bedenkenlos gelöscht werden (abgebrochene Schreibvorgänge) |
| Leerer Transkript-Block | Enthält nur Musik/Rauschen ohne Sprache (normal, wird trotzdem als erledigt markiert) |

## Lizenz

MIT – frei für private und kommerzielle Nutzung.
