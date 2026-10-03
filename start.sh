#!/usr/bin/env bash
#
# start.sh  -  Whisper-Aggregator-CLI Starter
#
# Transkribiert alle MP3s unter INPUT_PATH nach TRANSCRIPT_DIR
# (resume-fähig: schon vorhandene Transkripte werden übersprungen)
# und schreibt zusätzlich die Aggregat-Datei OUTPUT.
#
# Aufruf:
#   ./start.sh                      # mit den Standard-Werten unten
#   ./start.sh --overwrite          # alle Transkripte neu erzeugen
#   ./start.sh --model-size base    # andere Modellgröße
#   ./start.sh --help               # alle Optionen des Python-Tools
#
# Konfiguration per Umgebungsvariable, z. B.:
#   MODEL_SIZE=small DEVICE=cuda ./start.sh
#   WHISPER_LANGUAGE=de ./start.sh        # Sprache erzwingen
#
set -euo pipefail

# ---------------------------------------------------------------- Konfiguration
INPUT_PATH="${INPUT_PATH:-PodcastBulkDownloader/mp3}"   # Root-Verzeichnis (MP3s)
TRANSCRIPT_DIR="${TRANSCRIPT_DIR:-transcript}"          # Einzeltranskripte (.txt)
OUTPUT="${OUTPUT:-ergebnis.txt}"                        # Aggregat-Datei
MODEL_SIZE="${MODEL_SIZE:-tiny}"                        # tiny|base|small|medium|large|turbo
DEVICE="${DEVICE:-cpu}"                                 # cpu|cuda
WHISPER_LANGUAGE="${WHISPER_LANGUAGE:-}"                # z. B. "de" | leer = Auto-Erkennung
LOGFILE="${LOGFILE:-whisper_run.log}"                   # Lauf-Protokoll
# Sprachcode normalisieren: "de_DE" / "de-DE" -> "de" (Whisper erwartet "de")
[ -n "$WHISPER_LANGUAGE" ] && WHISPER_LANGUAGE="${WHISPER_LANGUAGE%%[_-]*}"

# ----------------------------------------------------------------Hilfsmittel
c_ok=""; c_warn=""; c_err=""; c_off=""
if [ -t 1 ]; then
    c_ok=$'\033[32m'; c_warn=$'\033[33m'; c_err=$'\033[31m'; c_off=$'\033[0m'
fi

info()  { printf '%s\n' "$*"; }
fail()  {
    local msg="$1" code="${2:-2}"
    printf '%sFEHLER:%s %s\n' "$c_err" "$c_off" "$msg" >&2
    exit "$code"
}

# ---------------------------------------------------------------- Verzeichnis
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------- Checks
command -v python3 >/dev/null 2>&1 \
    || fail "python3 nicht gefunden – bitte Python 3.9+ installieren."

python3 -c "import faster_whisper" >/dev/null 2>&1 \
    || fail "Abhängigkeiten fehlen – ausführen: pip install -r requirements.txt" 3

[ -d "$INPUT_PATH" ] \
    || fail "Eingabeverzeichnis nicht gefunden: $INPUT_PATH" 2

# ---------------------------------------------------------------- Resume-Info
total=$(find "$INPUT_PATH" -type f -iname '*.mp3' 2>/dev/null | wc -l | tr -d ' ')
done_files=0
if [ -d "$TRANSCRIPT_DIR" ]; then
    done_files=$(find "$TRANSCRIPT_DIR" -type f -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')
fi
todo=$(( total - done_files ))
[ "$todo" -lt 0 ] && todo=0

info "=================================================="
info " Whisper-Aggregator-CLI"
info "=================================================="
info "  Eingabe         : $INPUT_PATH"
info "  Transkripte     : $TRANSCRIPT_DIR"
info "  Aggregat        : $OUTPUT"
info "  Modell / Gerät  : $MODEL_SIZE / $DEVICE"
[ -n "$WHISPER_LANGUAGE" ] && info "  Sprache         : $WHISPER_LANGUAGE"
info "  MP3s gesamt     : $total"
if [ "$done_files" -gt 0 ]; then
    info "  ${c_ok}bereits fertig : $done_files (werden übersprungen)${c_off}"
    info "  zu verarbeiten  : $todo"
else
    info "  zu verarbeiten  : $total"
fi
info "  Protokoll       : $LOGFILE"
info "=================================================="
info ""

if [ "$total" -eq 0 ]; then
    fail "Keine MP3-Dateien unter: $INPUT_PATH"
fi

# ---------------------------------------------------------------- Argumente
args=(
    --path            "$INPUT_PATH"
    --transcript-dir  "$TRANSCRIPT_DIR"
    --output          "$OUTPUT"
    --model-size      "$MODEL_SIZE"
    --device          "$DEVICE"
)
[ -n "$WHISPER_LANGUAGE" ] && args+=( --language "$WHISPER_LANGUAGE" )

# zusätzliche Optionen direkt durchreichen, z. B. --overwrite / --no-aggregate
args+=( "$@" )

# ---------------------------------------------------------------- Lauf
set +e
python3 whisper_aggregator.py "${args[@]}" 2>&1 | tee "$LOGFILE"
status=${PIPESTATUS[0]}
set -e

info ""
case "$status" in
    0) info "${c_ok}Fertig – alle Transkripte vorhanden.${c_off}" ;;
    1) info "${c_warn}Teilweise Fehler – fehlgeschlagene Dateien werden beim nächsten Lauf erneut versucht.${c_off}" ;;
    *) info "${c_err}Abgebrochen (Exit-Code $status).${c_off}" ;;
esac
info "Erneut starten mit './start.sh' – es wird an der letzten Stelle weitergemacht."

exit "$status"
