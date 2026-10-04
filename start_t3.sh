#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start_t3.sh - Testlauf OHNE Sprechererkennung UND OHNE Timecodes
# (Variante von start_t3_sptc.sh: Diarization und Timecodes fest AUS)
#
# Ausgabe pro Segment:
#   Good morning. I'm Christina Karish...
#
# Standard (ohne Argumente):
#   - scannt rekursiv /content/PodcastBulkDownloader/mp3 (11 Folgen)
#   - Modell: turbo, Gerät: auto -> T4 + float16 (sonst CPU + int8)
#   - Zielausgabe: /content/transcript_t3  +  /content/ergebnis_t3.txt
#     (andere Ordner als die *_diar-Varianten - Resumes bleiben getrennt)
#   - Timecodes AUS (--no-timestamps explizit)
#   - Sprecher AUS (kein --enable-diarization, kein HF-Token noetig)
#
# Aufrufe:
#   ./start_t3.sh                                    # voller Lauf
#   ./start_t3.sh --overwrite                        # alles neu erzeugen
#   ./start_t3.sh --path "/content/.../CQ Morning Briefing"   # 1 Folge
#   DEVICE=cpu ./start_t3.sh                         # GPU ignorieren
#   DEVICE=cuda ./start_t3.sh                        # GPU erzwingen
#   EXTRA_ARGS="--beam-size 1 --language de" ./start_t3.sh
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/whisper_aggregator_t3.py"
PY="${PYTHON:-python3}"

# --- Konfiguration (jede Variable per Umgebung überschreibbar) -----------
MP3_DIR="${MP3_DIR:-/content/PodcastBulkDownloader/mp3}"
MODEL="${MODEL:-turbo}"
DEVICE="${DEVICE:-auto}"   # auto (T4+float16, sonst CPU+int8) | cuda | cpu

if [[ ! -f "$SCRIPT" ]]; then
  echo "FEHLER: Skript nicht gefunden: $SCRIPT" >&2
  exit 2
fi

# Zielordner: eigene, damit die *_diar-Läufe nicht vermischt werden
DEFAULT_TD="/content/transcript_t3"
DEFAULT_OUT="/content/ergebnis_t3.txt"
TRANSCRIPT_DIR="${TRANSCRIPT_DIR:-$DEFAULT_TD}"
OUTPUT="${OUTPUT:-$DEFAULT_OUT}"

# Effektive Werte: argparse nimmt bei doppelten Flags den letzten Wert
EFF_PATH="$MP3_DIR"
EFF_TD="$TRANSCRIPT_DIR"
EFF_OUT="$OUTPUT"
prev=""
for a in "$@"; do
  case "$prev" in
    --path)           EFF_PATH="$a" ;;
    --transcript-dir) EFF_TD="$a" ;;
    --output)         EFF_OUT="$a" ;;
  esac
  case "$a" in
    --path=*)           EFF_PATH="${a#--path=}" ;;
    --transcript-dir=*) EFF_TD="${a#--transcript-dir=}" ;;
    --output=*)         EFF_OUT="${a#--output=}" ;;
  esac
  prev="$a"
done

# --- Header ---------------------------------------------------------------
echo "=================================================="
echo " whisper_aggregator_t3.py - ohne Sprecher, ohne Timecodes"
echo "=================================================="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/ GPU   : /' || true
else
  echo " GPU   : nvidia-smi nicht gefunden (Läuft auf CPU)"
fi
echo " MP3   : $EFF_PATH"
echo " Ausgabe: $EFF_TD  +  $EFF_OUT"
echo " Modell : $MODEL | Gerät: $DEVICE | Timecodes: aus | Sprecher: aus"
echo "--------------------------------------------------"

# --- Argumente zusammenbauten ---------------------------------------------
ARGS=(
  --path "$MP3_DIR"
  --transcript-dir "$TRANSCRIPT_DIR"
  --output "$OUTPUT"
  --model-size "$MODEL"
  --device "$DEVICE"
  --no-timestamps
)

# Zusätzliche Flags aus EXTRA_ARGS (bewusstes Wortsplitting)
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

echo " Befehl: ${PY} whisper_aggregator_t3.py ${ARGS[*]} $*"
echo "=================================================="
echo

# Zusätzliche Argumente durchreichen ("$@") - bei doppelten Flags gewinnt
# der letzte Wert (z.B. "--no-timestamps ... --timestamps" = Timecodes an).
START=$(date +%s)
set +e
"$PY" -u "$SCRIPT" "${ARGS[@]}" "$@"
RC=$?
set -e
END=$(date +%s)

echo
echo "=================================================="
echo " Lauf beendet: Exit-Code $RC | Dauer: $((END - START)) s"
echo " Transkripte : $EFF_TD"
echo " Aggregat    : $EFF_OUT"
case "$RC" in
  0) echo " Ergebnis: reiner Text pro Segment" ;;
  1) echo " (1 = einzelne Dateien fehlgeschlagen - erneuter Lauf setzt fort)" ;;
esac
echo "=================================================="
exit $RC
