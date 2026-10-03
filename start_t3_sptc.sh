#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start_t3_sptc.sh - Testlauf MIT Sprechererkennung UND Timecodes
# (Variante von start.sh, nur Diarization-Modus fest auf AN)
#
# Ausgabe pro Segment:
#   [0.00s - 5.02s] SPEAKER_00: Good morning. I'm Christina Karish...
#
# Standard (ohne Argumente):
#   - scannt rekursiv /content/PodcastBulkDownloader/mp3 (11 Folgen)
#   - Modell: turbo, Gerät: auto -> T4 + float16
#   - Zielausgabe: /content/transcript_t3_diar  +  /content/ergebnis_t3_diar.txt
#     (gleiche Ordner wie "start.sh mit DIARIZATION=1" - ein Lauf des anderen
#      Skripts überspringt per Resume, es wird nichts doppelt transkribiert)
#   - Timecodes IMMER an (--timestamps wird explizit mitgegeben)
#
# Voraussetzung (einmalig):
#   1) Modell freischalten: https://huggingface.co/pyannote/speaker-diarization-3.1 -> "Agree"
#   2) Token-Datei erzeugen - in einer Zelle der Colab-UI ausführen:
#        from google.colab import userdata
#        open('/content/.hf_token','w').write(userdata.get('HF_TOKEN'))
#        import os; os.chmod('/content/.hf_token', 0o600)
#
# Aufrufe:
#   ./start_t3_sptc.sh                                    # voller Lauf
#   ./start_t3_sptc.sh --overwrite                        # alles neu erzeugen
#   ./start_t3_sptc.sh --path "/content/.../CQ Morning Briefing"   # 1 Folge
#   TOKEN_FILE=/pfad/zum/token ./start_t3_sptc.sh          # anderen Token nutzen
#   DIARIZATION=0 ./start_t3_sptc.sh                       # doch ohne Sprecher
#   EXTRA_ARGS="--beam-size 1 --language de" ./start_t3_sptc.sh
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/whisper_aggregator_t3.py"
PY="${PYTHON:-python3}"

# --- Konfiguration (jede Variable per Umgebung überschreibbar) -----------
MP3_DIR="${MP3_DIR:-/content/PodcastBulkDownloader/mp3}"
MODEL="${MODEL:-turbo}"
TOKEN_FILE="${TOKEN_FILE:-/content/.hf_token}"
DIARIZATION="${DIARIZATION:-1}"   # 1 (Default) | auto | 0

if [[ ! -f "$SCRIPT" ]]; then
  echo "FEHLER: Skript nicht gefunden: $SCRIPT" >&2
  exit 2
fi

# --- Diarization-Modus festlegen ------------------------------------------
case "$DIARIZATION" in
  1|on|ON|true|TRUE)     DIARI_ON=1 ;;
  0|off|OFF|false|FALSE) DIARI_ON=0 ;;
  auto) if [[ -f "$TOKEN_FILE" ]]; then DIARI_ON=1; else DIARI_ON=0; fi ;;
  *)
    echo "FEHLER: DIARIZATION muss 1, 0 oder auto sein (war: $DIARIZATION)" >&2
    exit 2
    ;;
esac

# Zielordner: gleiche wie "start.sh mit DIARIZATION=1" (Resume bleibt konsistent)
DEFAULT_TD="/content/transcript_t3_diar"
DEFAULT_OUT="/content/ergebnis_t3_diar.txt"
TRANSCRIPT_DIR="${TRANSCRIPT_DIR:-$DEFAULT_TD}"
OUTPUT="${OUTPUT:-$DEFAULT_OUT}"

# Effektive Werte: argparse nimmt bei doppelten Flags den letzten Wert
EFF_PATH="$MP3_DIR"
EFF_TD="$TRANSCRIPT_DIR"
EFF_OUT="$OUTPUT"
TOKEN_FLAG_GIVEN=0
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
    --hf-token|--hf-token-file) TOKEN_FLAG_GIVEN=1 ;;
    --hf-token=*|--hf-token-file=*) TOKEN_FLAG_GIVEN=1 ;;
  esac
  prev="$a"
done

# --- Header ---------------------------------------------------------------
echo "=================================================="
echo " whisper_aggregator_t3.py - Sprecher + Timecodes"
echo "=================================================="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/ GPU   : /' || true
else
  echo " GPU   : nvidia-smi nicht gefunden (Läuft auf CPU)"
fi
echo " MP3   : $EFF_PATH"
echo " Ausgabe: $EFF_TD  +  $EFF_OUT"
echo " Modell : $MODEL | Gerät: auto | Timecodes: an | Sprecher: an"
echo "--------------------------------------------------"

# --- Vorab-Prüfung: Token da? --------------------------------------------
# (Verhindert einen Lauf, der erst nach dem Modell-Download mit 6 abbricht;
#  ein mitgegebenes --hf-token/--hf-token-file überspringt den Check.)
if [[ "$DIARI_ON" == "1" && "$TOKEN_FLAG_GIVEN" == "0" ]]; then
  if [[ ! -f "$TOKEN_FILE" ]]; then
    cat >&2 <<EOF
FEHLER: Token-Datei '$TOKEN_FILE' fehlt - ohne sie kann keine
Sprecher-Erkennung laufen (Exit-Code 6).

In einer Zelle der Colab-UI ausführen (wichtig: aus der UI, nicht aus der Shell):

    from google.colab import userdata
    import os
    open('$TOKEN_FILE','w').write(userdata.get('HF_TOKEN'))
    os.chmod('$TOKEN_FILE', 0o600)
    print("geschrieben:", os.path.getsize('$TOKEN_FILE'), "Bytes")

Zusätzlich das Modell einmalig freischalten:
    https://huggingface.co/pyannote/speaker-diarization-3.1  -> "Agree"

Alternativ: TOKEN_FILE=<anderer Pfad> ./start_t3_sptc.sh
        oder: ./start_t3_sptc.sh --hf-token hf_xxx
EOF
    exit 6
  fi
  if [[ ! -s "$TOKEN_FILE" ]]; then
    echo "FEHLER: Token-Datei '$TOKEN_FILE' ist leer (Exit-Code 6)." >&2
    exit 6
  fi
fi

# --- Argumente zusammenbaren ---------------------------------------------
ARGS=(
  --path "$MP3_DIR"
  --transcript-dir "$TRANSCRIPT_DIR"
  --output "$OUTPUT"
  --model-size "$MODEL"
)

if [[ "$DIARI_ON" == "1" ]]; then
  # --timestamps steht hier bewusst explizit dabei, auch wenn das Skript die
  # Timecodes bei Diarization bereits selbst einschaltet.
  ARGS+=(--enable-diarization --timestamps)
  if [[ "$TOKEN_FLAG_GIVEN" == "0" ]]; then
    ARGS+=(--hf-token-file "$TOKEN_FILE")
  fi
fi

# Zusätzliche Flags aus EXTRA_ARGS (bewusstes Wortsplitting)
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

echo " Befehl: ${PY} whisper_aggregator_t3.py ${ARGS[*]} $*"
echo "=================================================="
echo

# Zusätzliche Argumente durchreichen ("$@") - bei doppelten Flags gewinnt
# der letzte Wert.
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
  0) echo " Ergebnis: [start - end] SPEAKER_xx: Text" ;;
  1) echo " (1 = einzelne Dateien fehlgeschlagen - erneuter Lauf setzt fort)" ;;
  6) echo " (6 = Diarization prüfen: Token-Datei vorhanden? Modell auf HF freigeschaltet?)" ;;
esac
echo "=================================================="
exit $RC
