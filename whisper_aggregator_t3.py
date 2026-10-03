#!/usr/bin/env python3
"""Whisper-Aggregator-CLI

Scannt ein Verzeichnis rekursiv nach MP3-Dateien, transkribiert jede Datei
mit faster-whisper (GPU-first: Nvidia Tesla T4 / CUDA, float16) und schreibt:

  1. je MP3 eine EINZELNE Textdatei in das Transcript-Verzeichnis
     (Verzeichnisstruktur 1:1 gespiegelt) -> RESUME-FÄHIG
  2. optional alle Transkripte aggregiert in eine globale Textdatei

Resume-Logik: Vorhandene Transcript-Dateien werden übersprungen. Durch
atomares Schreiben (Schreiben auf .part + rename) kann ein abgebrochener
Lauf jederzeit neu gestartet werden und fährt an der letzten Stelle fort.

Aufruf-Beispiel (GPU, automatisch erkannt):
    python whisper_aggregator_t3.py --path ./mp3 --transcript-dir ./transcript
    (oder bequem: ./start.sh)

Gerätewahl:
    --device auto  (Standard)  -> CUDA/GPU, wenn eine GPU da ist, sonst CPU
    --device cuda               -> erzwingt GPU (Fehler, falls keine da ist)
    --device cpu                -> erzwingt CPU

Optionale Annotation (Opt-in, standardmäßig AUS):
    --timestamps                -> Timecodes pro Segment: [0.00s - 4.52s] Text
    --enable-diarization        -> Sprecher-Erkennung via PyAnnote:
                                   [0.00s - 4.52s] SPEAKER_00: Text
                                   (setzt Timecodes automatisch an,
                                   Abschalten mit --no-timestamps)
    HF-Token für die gated PyAnnote-Modelle, Reihenfolge:
    --hf-token > --hf-token-file > Umgebungsvariable HF_TOKEN > Colab-Secret
    (letzteres nur aus einer Zelle, die aus der Colab-UI gestartet wurde)

    Colab-Brücke für das Secret (in einer Notebook-Zelle ausführen):
        from google.colab import userdata
        open('/content/.hf_token', 'w').write(userdata.get('HF_TOKEN'))
    und dann:  ... --enable-diarization --hf-token-file /content/.hf_token

GPU-Hinweise (z.B. Nvidia Tesla T4):
  * Standard-Rechentyp auf GPU ist float16 -> auf der T4 der schnellste Modus.
  * Modell 'turbo' (large-v3-turbo) ist auf der T4 die beste Wahl für
    Podcasts: large-v3-Qualität bei ca. 20x Echtzeit-Durchsatz.
  * Beim Start werden zwei typische Container-/venv-Fallstricke automatisch
    entschärft: fehlende CUDA-12-Laufzeitbibliotheken (libcublas.so.12) und
    das in PyAV >= 19 entfernte av.open(metadata_errors=...)-Argument.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # tqdm ist optional -> Fallback-Wrapper
    tqdm = None  # type: ignore[assignment]

# Erlaubte Whisper-Modellgrößen (PRD: tiny, base, small, medium, large)
MODEL_SIZES = (
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v3",
    "large-v3-turbo",
    "turbo",
)
# "auto" = CUDA wenn eine GPU da ist, sonst CPU
DEVICES = ("auto", "cpu", "cuda")
COMPUTE_TYPES = {"cpu": "int8", "cuda": "float16"}

SEPARATOR_TOP = "--- FILE: {name} ---"
SEPARATOR_BOTTOM = "-" * 42

# Suffix für atomare Zwischendateien
PART_SUFFIX = ".part"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="whisper-aggregator",
        description=(
            "Durchsucht ein Verzeichnis rekursiv nach MP3-Dateien, "
            "transkribiert sie mit faster-whisper, schreibt je Datei eine "
            "eigene Textdatei ins Transcript-Verzeichnis (resume-fähig) und "
            "aggregiert alle Ergebnisse optional in einer Textdatei."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Pfad zum Root-Verzeichnis (Input), wird rekursiv durchsucht.",
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=Path("transcript"),
        help="Verzeichnis für die EINZELNEN Transkripte (.txt), "
        "Verzeichnisstruktur des Inputs wird 1:1 gespiegelt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ergebnis.txt"),
        help="Name/Pfad der aggregierten Ziel-Textdatei.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Keine aggregierte Datei schreiben, nur die Einzeltranskripte.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Vorhandene Transkripte erneut erzeugen (Resume deaktivieren).",
    )
    parser.add_argument(
        "--model-size",
        choices=MODEL_SIZES,
        default="turbo",
        help="Whisper-Modellgröße. 'turbo' (large-v3-turbo) ist die "
        "Empfehlung für GPU/T4: large-v3-Qualität bei ca. 20x Echtzeit.",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="auto",
        help="Ausführungsgerät: 'auto' wählt GPU (CUDA), falls vorhanden, "
        "sonst CPU. 'cuda' erzwingt die GPU.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="GPU-Index bei mehreren GPUs (nur --device cuda/auto).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam-Search-Breite. 1 = schnellste auf GPU, 5 = beste Qualität.",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Override für den Rechentyp (z.B. int8, float16, float32). "
        "Standard: float16 (GPU) bzw. int8 (CPU). Auf der T4 ist float16 "
        "der schnellste Typ.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optionale Sprachangabe (z.B. 'de', 'en'). Standard: Auto-Erkennung.",
    )
    # --- Optionale Annotation: Timecodes + Sprecher-Erkennung ------------
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "--timestamps",
        "--timecodes",
        dest="timestamps",
        action="store_true",
        help="Timecodes pro Segment ergänzen, z.B. '[0.00s - 4.52s] Text'. "
        "Standard: an bei --enable-diarization, sonst aus.",
    )
    time_group.add_argument(
        "--no-timestamps",
        dest="timestamps",
        action="store_false",
        help="Timecodes unterdrücken (auch bei --enable-diarization).",
    )
    parser.set_defaults(timestamps=None)
    parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="Sprecher-Erkennung (PyAnnote) zusätzlich zur Transkription. "
        "Braucht ein HF-Token und die Modell-Freischaltung auf huggingface.co.",
    )
    parser.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-3.1",
        help="HuggingFace-Checkpoint der Diarization-Pipeline.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace-Token (read). Ohne diesen Argument werden "
        "--hf-token-file, Umgebungsvariable HF_TOKEN bzw. das Colab-Secret "
        "HF_TOKEN geprüft.",
    )
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        default=None,
        help="Datei, die den HuggingFace-Token enthält (Empfehlung, wenn das "
        "Token aus einem Colab-Secret kommt: dort per Zelle in eine Datei "
        "schreiben). Bevorzugt vor Umgebungsvariable/Colab-Secret geprüft.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# File Search (pathlib)
# --------------------------------------------------------------------------
def find_mp3_files(root: Path) -> List[Path]:
    """Sucht rekursiv alle *.mp3 Dateien unterhalb von `root`."""
    if not root.exists():
        raise FileNotFoundError(f"Eingabeverzeichnis nicht gefunden: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"--path muss ein Verzeichnis sein: {root}")

    files = sorted(
        (p for p in root.glob("**/*.mp3") if p.is_file()),
        key=lambda p: str(p).lower(),
    )
    # Zusätzlich .MP3/.Mp3 auf Gross-Kleinschreibung-sensitiven Dateisystemen
    seen = {p.resolve() for p in files}
    extra = (
        p
        for p in root.glob("**/*.MP3")
        if p.is_file() and p.resolve() not in seen
    )
    files.extend(sorted(extra, key=lambda p: str(p).lower()))
    return files


def transcript_path(root: Path, transcript_dir: Path, mp3: Path) -> Path:
    """Ziel-Pfad für ein Einzeltranskript (Struktur 1:1, Suffix .txt)."""
    rel = mp3.relative_to(root)
    return transcript_dir / rel.with_suffix(".txt")


def relative_name(root: Path, mp3: Path) -> str:
    """Anzeige-Name relativ zum Root (Dateiname, wenn direkt darunter)."""
    rel = mp3.relative_to(root)
    return str(rel) if rel.parent != Path(".") else rel.name


# --------------------------------------------------------------------------
# Atomic Write (Basis für Resume)
# --------------------------------------------------------------------------
def write_text_atomic(path: Path, text: str) -> None:
    """Schreibt atomar: erst *.part, dann rename -> Datei ist entweder
    komplett da oder gar nicht. Ein Abbruch hinterlässt nie ein
    halbgeschriebenes Transkript, das beim Resume für 'fertig' gehalten wird."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + PART_SUFFIX)
    with part.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(part, path)


def plan_files(
    root: Path, transcript_dir: Path, overwrite: bool
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Liefert (todo, already_done) als Liste von (mp3, ziel_txt)."""
    todo: List[Tuple[Path, Path]] = []
    done: List[Tuple[Path, Path]] = []
    for mp3 in find_mp3_files(root):
        target = transcript_path(root, transcript_dir, mp3)
        if not overwrite and target.exists():
            done.append((mp3, target))
        else:
            todo.append((mp3, target))
    return todo, done


# --------------------------------------------------------------------------
# Transcription Engine
# --------------------------------------------------------------------------
def cuda_device_count() -> int:
    """Anzahl der per ctranslate2 sichtbaren CUDA-GPUs (0 = keine)."""
    try:
        import ctranslate2
    except ImportError:
        return 0
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def resolve_device(requested: str) -> str:
    """'auto' -> GPU, wenn da; 'cuda' -> erzwingt GPU, sonst Fehler."""
    have_gpu = cuda_device_count() > 0
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not have_gpu:
            raise LookupError(
                "--device cuda angegeben, aber es wurde keine CUDA-GPU "
                "erkannt (ctranslate2 meldet 0 Geräte).\n"
                "  -> Treiber/GPU prüfen: nvidia-smi\n"
                "  -> CUDA-Container: CUDA_VISIBLE_DEVICES darf die GPU "
                "nicht ausblenden\n"
                "  -> oder mit --device auto bzw. --device cpu starten."
            )
        return "cuda"
    return "cuda" if have_gpu else "cpu"


def describe_gpu(index: int) -> str:
    """Kurzbeschreibung der GPU (Name + VRAM), via pynvml/nvidia-smi."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = mem.total / 1024**3
            free = mem.free / 1024**3
            return f"{name} (GPU {index}, {total:.0f} GiB VRAM, {free:.1f} GiB frei)"
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return f"GPU {index}"


def ensure_pyav_compat() -> None:
    """PyAV >= 19 entfernte das Argument `metadata_errors` aus av.open(),
    das faster-whisper beim MP3-Decode noch mitgibt. Ohne diese Brücke
    scheitert jede Datei mit
        TypeError: open() got an unexpected keyword argument 'metadata_errors'

    av.open() ist eine Cython-Builtin-Funktion (keine Python-Signatur),
    deshalb wird der Aufruf nicht statisch, sondern laufzeitlich abgefangen:
    Fällt genau dieser TypeError, wird ohne das Argument wiederholt - bei
    PyAV-Versionen, die es kennen, passiert nichts."""
    try:
        import av
    except ImportError:
        return  # PyAV fehlt komplett -> Fehler kommt später ohnehin klarer

    if getattr(av.open, "_whisper_aggregator_compat", False):
        return  # bereits umgehängt

    original_open = av.open

    def open_compat(*args, **kwargs):
        try:
            return original_open(*args, **kwargs)
        except TypeError as exc:
            if "metadata_errors" not in kwargs or "metadata_errors" not in str(exc):
                raise
            kwargs.pop("metadata_errors")
            return original_open(*args, **kwargs)

    open_compat.__name__ = "open"
    open_compat.__doc__ = getattr(original_open, "__doc__", None)
    open_compat._whisper_aggregator_compat = True
    av.open = open_compat  # faster_whisper.audio greift av.open zur Laufzeit auf


def _site_package_dirs() -> List[str]:
    """Alle Site-Pakete-Verzeichnisse (System, User, sys.path)."""
    import site as site_mod

    dirs: List[str] = []
    try:
        dirs.extend(site_mod.getsitepackages())
    except AttributeError:  # z.B. in manchen Virtualenvs
        pass
    try:
        dirs.append(site_mod.getusersitepackages())
    except Exception:
        pass
    dirs.extend(
        p
        for p in sys.path
        if p and p.rstrip("/").endswith(("site-packages", "dist-packages"))
    )
    return list(dict.fromkeys(d for d in dirs if d))


def ensure_cuda_runtime() -> List[str]:
    """ctranslate2 (CUDA-12-Build) lädt libcublas.so.12 & Co. per dlopen.
    In Python-Umgebungen liegen diese Bibliotheken in
    <site-packages>/nvidia/*/lib und fehlen im ldconfig-Cache -> sonst:
        Library libcublas.so.12 is not found or cannot be loaded
    Wir finden die NVIDIA-Pip-Pakete, hängen die Pfade an LD_LIBRARY_PATH
    und preloaden die Bibliotheken, damit der ctranslate2-Laufzeitlinker
    sie findet. Reihenfolge: Basis -> abhängige Bibliotheken.
    Gibt die tatsächlich geladenen Pfade zurück (leer = nichts gefunden)."""
    import ctypes
    import glob

    wanted = [
        "libcudart.so.12",
        "libnvJitLink.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcufft.so.11",
        "libcurand.so.10",
        "libcusolver.so.11",
        "libcusparse.so.12",
        "libcupti.so.12",
    ]
    search_roots = [
        os.path.join(d, "nvidia")
        for d in _site_package_dirs()
    ]

    loaded: List[str] = []
    for lib_name in wanted:
        for root in search_roots:
            match = sorted(glob.glob(os.path.join(root, "*", "lib", lib_name)))
            if not match:
                continue
            path = match[0]
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                loaded.append(path)
                break
            except OSError:
                continue  # inkompatibel/unvollständig -> nächster Treffer

    if loaded:
        lib_dirs = list(dict.fromkeys(os.path.dirname(p) for p in loaded))
        existing = os.environ.get("LD_LIBRARY_PATH", "").split(":")
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            [*lib_dirs, *[d for d in existing if d]]
        )
    return loaded


def load_model(model_size: str, device: str, compute_type: str, device_index: int = 0):
    """Initialisiert das faster-whisper Modell (CPU/GPU)."""
    ensure_pyav_compat()
    if device == "cuda":
        cuda_libs = ensure_cuda_runtime()
        if cuda_libs:
            print(
                f"CUDA-Laufzeit bereitgestellt: {len(cuda_libs)} NVIDIA-"
                f"Bibliothek(en) geladen (u.a. {os.path.basename(cuda_libs[-1])})"
            )
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "FEHLER: 'faster-whisper' ist nicht installiert.\n"
            "  Installation: pip install faster-whisper"
        ) from exc

    target = device.upper()
    if device == "cuda":
        target = f"{target} ({describe_gpu(device_index)})"

    print(
        f"Lade Whisper-Modell '{model_size}' auf {target} "
        f"(compute_type={compute_type}) ..."
    )
    return WhisperModel(
        model_size,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
    )


def get_hf_token(
    explicit: Optional[str] = None, token_file: Optional[Path] = None
) -> Tuple[Optional[str], str]:
    """Sucht den HuggingFace-Token für die gated PyAnnote-Modelle.

    Reihenfolge: --hf-token > --hf-token-file > Umgebungsvariable
    (HF_TOKEN/HUGGING_FACE_HUB_TOKEN) > Colab-Secret (google.colab.userdata).

    Der Colab-Secret-Weg funktioniert nur aus einer Zelle, die aus der
    Colab-UI gestartet wurde - ein Abruf aus einem reinen Shell-Prozess
    läuft in einen Timeout. Deshalb ist --hf-token-file der robuste Weg:
    Zelle schreibt das Secret in eine Datei, dieses Skript liest sie.

    Gibt (Token, Quelle) zurück; Token ist None, wenn nirgends einer ist."""
    if explicit and explicit.strip():
        return explicit.strip(), "--hf-token"

    if token_file is not None:
        path = token_file.expanduser()
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise DiarizationError(
                f"Token-Datei '{path}' nicht lesbar: {exc}"
            ) from exc
        # erste nicht-leere Zeile, sonst ignorieren
        for line in content.splitlines():
            if line.strip():
                return line.strip(), f"Token-Datei {path}"
        raise DiarizationError(f"Token-Datei '{path}' ist leer.")

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"):
        value = os.environ.get(var)
        if value and value.strip():
            return value.strip(), f"Umgebungsvariable {var}"

    # Colab: `from google.colab import userdata; userdata.get('HF_TOKEN')`
    # schlägt außerhalb eines echten Colab-Kernels fehl -> darf nie crashen.
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None, "kein Token gefunden"
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        try:
            value = userdata.get(key)
        except Exception:
            continue
        if value and str(value).strip():
            return str(value).strip(), f"Colab-Secret {key}"
    return None, "kein Token gefunden"


def apply_torchaudio_patches() -> None:
    """Kompatibilitäts-Patches für Python 3.13 / torchaudio 2.x + pyannote.

    torchaudio 2.11 kennt weder `AudioMetaData` noch `list_audio_backends`
    mehr, pyannote.audio erwartet aber beides. Fehlen die Attribute, wird
    nachgeholfen - Attribute, die schon da sind, bleiben unangetastet."""
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        audio_meta_data = None
        for module_name in ("torchaudio._backend", "torchaudio._backend.common"):
            try:
                module = __import__(module_name, fromlist=["AudioMetaData"])
                audio_meta_data = getattr(module, "AudioMetaData")
                break
            except (ImportError, AttributeError):
                continue
        if audio_meta_data is None:
            from dataclasses import dataclass

            @dataclass
            class AudioMetaData:  # Minimal-Stub, nur für Typ-/isinstance-Checks
                sample_rate: int
                num_frames: int
                num_channels: int
                bits_per_sample: int
                encoding: str

            audio_meta_data = AudioMetaData
        torchaudio.AudioMetaData = audio_meta_data

    if not hasattr(torchaudio, "list_audio_backends"):
        def _list_audio_backends() -> List[str]:
            return ["ffmpeg", "soundfile"]

        torchaudio.list_audio_backends = _list_audio_backends


class DiarizationError(RuntimeError):
    """Konfigurations-/Zugangsfehler bei der Sprecher-Erkennung (Exit-Code 6)."""


def load_diarization_pipeline(
    model_name: str, token: Optional[str], device: str
):
    """Lädt die PyAnnote-Diarization-Pipeline (ggf. auf der GPU).

    Reihenfolge ist wichtig: erst torchaudio patchen, DANN pyannote importieren."""
    apply_torchaudio_patches()

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationError(
            "'pyannote.audio' ist nicht installiert.\n"
            "  Installation: pip install pyannote.audio"
        ) from exc

    print(
        f"Lade Diarization-Pipeline '{model_name}' "
        f"(Token-Quelle: {'Token vorhanden' if token else 'kein Token'}) ..."
    )
    load_error: Optional[Exception] = None
    pipeline = None
    try:
        pipeline = Pipeline.from_pretrained(model_name, token=token)
    except TypeError:  # ältere pyannote-Versionen kennen kein token=
        try:
            pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        except Exception as exc:
            load_error = exc
    except Exception as exc:
        load_error = exc

    if pipeline is None:
        message = str(load_error)
        hint = ""
        if (
            "401" in message
            or "gated" in message.lower()
            or "restricted" in message.lower()
            or "token" in message.lower()
        ):
            hint = (
                f"\n  -> Modell ist auf huggingface.co GATED: Seite des "
                f"Modells '{model_name}' einmalig besuchen und den Zugang "
                "akzeptieren, danach ein Token bereitstellen.\n"
                "  -> Token-Quellen: --hf-token <token> | Umgebungsvariable "
                "HF_TOKEN | Colab-Secret 'HF_TOKEN'"
            )
        raise DiarizationError(
            f"Diarization-Pipeline nicht ladbar: {message}{hint}"
        ) from load_error

    try:
        import torch

        pipeline.to(torch.device("cuda" if device == "cuda" else "cpu"))
    except Exception as exc:  # CPU-Restlauf ist besser als gar nichts
        print(f"Hinweis: PyAnnote-Pipeline blieb auf dem Standardgerät ({exc})")
    return pipeline


def get_speaker_for_time(
    start_time: float, end_time: float, speaker_segments: List[dict]
) -> str:
    """Liefert den Sprecher, der im Zeitraum [start_time, end_time] den
    größten Anteil der Zeit geredet hat (Overlap-Matching)."""
    best_speaker = "UNKNOWN"
    max_overlap = 0.0
    for seg in speaker_segments:
        overlap_start = max(start_time, seg["start"])
        overlap_end = min(end_time, seg["end"])
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = seg["speaker"]
    return best_speaker


def run_diarization(
    diarization_pipeline, file_path: Path
) -> Tuple[List[dict], Optional[str]]:
    """Führt die Sprechertrennung aus. Rückgabe: (Sprecher-Segmente, Fehler).

    Fehler werden NICHT zum Abbruch: fehlt die Diarization, bleibt das
    Transkript trotzdem nutzbar (ohne Sprecher-Labels)."""
    if diarization_pipeline is None:
        return [], None
    try:
        diarization = diarization_pipeline(str(file_path))
        speaker_segments = [
            {"start": turn.start, "end": turn.end, "speaker": str(speaker)}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        return speaker_segments, None
    except Exception as exc:
        _warn_diarization_once(file_path, exc)
        return [], str(exc)


_DIARIZATION_WARN_COUNT = 0


def _warn_diarization_once(file_path: Path, exc: Exception) -> None:
    """Volle Fehlermeldung nur beim ersten Mal, danach Kurzform."""
    global _DIARIZATION_WARN_COUNT
    _DIARIZATION_WARN_COUNT += 1
    if _DIARIZATION_WARN_COUNT == 1:
        print(
            f"\n[WARN] Sprecher-Erkennung für '{file_path.name}' "
            f"fehlgeschlagen: {exc}",
            file=sys.stderr,
        )
        print(
            "[WARN] Lauf wird ohne Sprecher-Labels fortgesetzt "
            "(weitere Dateien nur noch kurz gemeldet).",
            file=sys.stderr,
        )
    else:
        print(
            f"\n[WARN] Diarization fehlgeschlagen: {file_path.name}",
            file=sys.stderr,
        )


def transcribe_file(
    model,
    diarization_pipeline,
    file_path: Path,
    language: Optional[str],
    beam_size: int = 5,
    timestamps: bool = False,
) -> Tuple[str, dict]:
    """Transkribiert eine Audiodatei und verheiratet Whisper mit PyAnnote.

    A) Sprechertrennung (falls Pipeline übergeben),
    B) Transkription mit Wort-Zeitstempeln,
    C) Formatierung je Segment: [start - end] SPEAKER_xx: Text

    Rückgabe: (Text, Kennzahlen) mit Kennzahlen = Audiodauer in Sekunden,
    benötigte Wall-Zeit, Durchsatz, Sprache und gefundene Sprecher.
    """
    started = time.perf_counter()

    # A) Sprechertrennung ausführen (falls Pipeline übergeben)
    speaker_segments, diarization_error = run_diarization(diarization_pipeline, file_path)

    # B) Transkription mit Wort-Zeitstempeln (für feine Zuordnung genutzt)
    segments, info = model.transcribe(
        str(file_path),
        language=language,
        beam_size=beam_size,
        vad_filter=True,  # Rausch-/Pausenfilter -> saubererer Turbo-Flow
        word_timestamps=bool(timestamps or diarization_pipeline is not None),
    )

    # C) Text + Timecode + Sprecher zusammenbauen
    lines: List[str] = []
    speaker_counts: dict = {}
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        parts: List[str] = []
        if timestamps:
            # Timecode formatiert, z.B. [0.00s - 4.52s]
            parts.append(f"[{seg.start:.2f}s - {seg.end:.2f}s]")
        if speaker_segments:
            speaker = get_speaker_for_time(seg.start, seg.end, speaker_segments)
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            parts.append(f"{speaker}:")
        parts.append(text)
        lines.append(" ".join(parts))

    text_output = "\n".join(lines)
    elapsed = time.perf_counter() - started

    audio_seconds = float(getattr(info, "duration", 0.0) or 0.0)
    speed = (audio_seconds / elapsed) if elapsed > 0 else 0.0
    stats = {
        "audio": audio_seconds,
        "elapsed": elapsed,
        "speed": speed,
        "language": getattr(info, "language", None) or (language or "-"),
        "speakers": sorted(speaker_counts),
        "diarization": bool(speaker_segments),
        "diarization_error": diarization_error,
    }
    return text_output, stats


# --------------------------------------------------------------------------
# Data Aggregation
# --------------------------------------------------------------------------
def format_block(name: str, transcript: str) -> str:
    """Formatiert einen Transkriptions-Block gemäß PRD-Vorgabe."""
    return (
        f"{SEPARATOR_TOP.format(name=name)}\n"
        f"{transcript}\n"
        f"{SEPARATOR_BOTTOM}\n"
    )


def build_aggregate(
    root: Path, items: List[Tuple[Path, Path]]
) -> str:
    """Liest alle vorhandenen Einzeltranskripte und baut den Aggregat-Text."""
    blocks: List[str] = []
    for _mp3, target in items:
        if not target.exists():
            continue  # fehlgeschlagene Datei -> nicht aufnehmen
        transcript = target.read_text(encoding="utf-8").strip()
        blocks.append(format_block(relative_name(root, _mp3), transcript))
    return "".join(blocks)


# --------------------------------------------------------------------------
# Progress (tqdm mit Fallback)
# --------------------------------------------------------------------------
def progress_bar(items):
    """Fortschrittsbalken über (mp3, ziel)-Tupel."""
    if tqdm is not None:
        return tqdm(items, desc="Transkribiere", unit="MP3", ncols=80)
    # Minimaler Fallback, falls tqdm fehlt
    class _Plain:
        def __init__(self, entries):
            self.entries = entries
            self.total = len(entries)

        def __iter__(self) -> Iterator[tuple]:
            for i, item in enumerate(self.entries, 1):
                mp3 = item[0] if isinstance(item, tuple) else item
                print(f"  [{i}/{self.total}] {mp3.name}")
                yield item

        def set_postfix_str(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    return _Plain(items)


# --------------------------------------------------------------------------
# Main Workflow
# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # 1. CLI Arguments parsen + Gerät auflösen ---------------------------
    root: Path = args.path.expanduser().resolve()
    transcript_dir: Path = args.transcript_dir.expanduser()
    output: Path = args.output.expanduser()

    try:
        device: str = resolve_device(args.device)
    except LookupError as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 5
    compute_type: str = args.compute_type or COMPUTE_TYPES[device]

    # 2. Alle *.mp3 Dateien finden + Resume-Planung ----------------------
    try:
        todo, already_done = plan_files(root, transcript_dir, args.overwrite)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 2

    total = len(todo) + len(already_done)
    if total == 0:
        print(f"Keine MP3-Dateien gefunden unter: {root}")
        return 0

    print(f"Gefunden: {total} MP3-Datei(en) unter {root}")
    gpu_note = f" ({describe_gpu(args.device_index)})" if device == "cuda" else ""
    # Timecodes: explizite Entscheidung gewinnt, sonst an bei Diarization
    use_timestamps: bool = (
        args.timestamps if args.timestamps is not None else args.enable_diarization
    )
    print(
        f"Gerät: {device.upper()}{gpu_note} | compute_type={compute_type} | "
        f"Modell: {args.model_size} | beam_size={args.beam_size} | "
        f"Timecodes: {'an' if use_timestamps else 'aus'} | "
        f"Diarization: {'an' if args.enable_diarization else 'aus'}"
    )
    if already_done:
        print(
            f"Resume: {len(already_done)} Transkript(e) vorhanden "
            f"in '{transcript_dir}' -> werden übersprungen "
            f"(mit --overwrite neu erzeugen)"
        )

    if not todo:
        print("Nichts zu tun – alle Transkripte sind bereits vorhanden.")
        skipped = len(already_done)
        failed: List[tuple] = []
        audio_total = 0.0
        work_total = 0.0
        languages = {}
        diarized_files = 0
        diarization_errors = 0
        speaker_total = 0
    else:
        # 3a. Diarization-Pipeline zuerst (-> Fehler früh, bevor Whisper lädt)
        diarization_pipeline = None
        if args.enable_diarization:
            try:
                hf_token, token_source = get_hf_token(
                    args.hf_token, args.hf_token_file
                )
            except DiarizationError as exc:
                print(f"FEHLER: {exc}", file=sys.stderr)
                return 6
            if not hf_token:
                print(
                    "FEHLER: --enable-diarization benötigt einen "
                    "HuggingFace-Token, aber keiner gefunden.\n"
                    "  -> Optionen: --hf-token <token> | --hf-token-file <file>"
                    " | Umgebungsvariable HF_TOKEN | Colab-Secret 'HF_TOKEN'\n"
                    f"  -> Modell '{args.diarization_model}' ist auf "
                    "huggingface.co zudem gated (Zugang dort akzeptieren).",
                    file=sys.stderr,
                )
                return 6
            print(f"HF-Token gefunden (Quelle: {token_source})")
            try:
                diarization_pipeline = load_diarization_pipeline(
                    args.diarization_model, hf_token, device
                )
            except DiarizationError as exc:
                print(f"FEHLER: {exc}", file=sys.stderr)
                return 6
            except Exception as exc:
                print(
                    f"FEHLER beim Laden der Diarization-Pipeline: {exc}",
                    file=sys.stderr,
                )
                return 6

        # 3. Whisper Modell initialisieren ------------------------------
        try:
            model = load_model(
                args.model_size, device, compute_type, args.device_index
            )
        except SystemExit:
            raise
        except Exception as exc:  # z.B. fehlende CUDA-Laufzeit
            print(f"FEHLER beim Laden des Modells: {exc}", file=sys.stderr)
            return 3

        # 4. Transkribieren -> EINZELDATEIEN (atomar geschrieben) -------
        skipped = len(already_done)
        failed = []
        done_now = 0
        audio_total = 0.0
        work_total = 0.0
        languages: dict = {}
        diarized_files = 0
        diarization_errors = 0
        speaker_total = 0
        bar = progress_bar(todo)

        for mp3, target in bar:
            try:
                text_content, stats = transcribe_file(
                    model,
                    diarization_pipeline,
                    mp3,
                    args.language,
                    args.beam_size,
                    use_timestamps,
                )
                write_text_atomic(target, text_content)
                done_now += 1
                audio_total += stats["audio"]
                work_total += stats["elapsed"]
                languages[stats["language"]] = languages.get(stats["language"], 0) + 1
                if stats["diarization"]:
                    diarized_files += 1
                    speaker_total += len(stats["speakers"])
                if stats["diarization_error"]:
                    diarization_errors += 1
                if tqdm is not None:
                    speed = (audio_total / work_total) if work_total > 0 else 0.0
                    extra = (
                        f" Sprecher={len(stats['speakers'])}"
                        if stats["diarization"]
                        else ""
                    )
                    bar.set_postfix_str(
                        f"ok={done_now} Fehler={len(failed)} "
                        f"überspr.{skipped} {speed:.1f}x Echtzeit{extra}"
                    )
            except Exception as exc:
                # Error Handling: NICHT abbrechen, sondern weitermachen.
                failed.append((str(mp3), str(exc)))
                print(
                    f"\n[FEHLER] Konnte '{mp3.name}' nicht transkribieren: "
                    f"{exc}",
                    file=sys.stderr,
                )
                if tqdm is not None:
                    bar.set_postfix_str(
                        f"ok={done_now} Fehler={len(failed)} "
                        f"überspr.{skipped}"
                    )
                continue

        if tqdm is not None:
            bar.close()

    # 5. Aggregat schreiben (auch aus bereits vorhandenen Transkripten) ---
    all_items = already_done + [
        (mp3, transcript_path(root, transcript_dir, mp3)) for mp3, _ in todo
    ]
    # Originalreihenfolge (sortiert) für konsistente Ausgabe
    all_items.sort(key=lambda pair: str(pair[0]).lower())

    if not args.no_aggregate:
        aggregate_text = build_aggregate(root, all_items)
        if aggregate_text:
            try:
                write_text_atomic(output, aggregate_text)
            except OSError as exc:
                print(
                    f"FEHLER beim Schreiben der Ausgabedatei: {exc}",
                    file=sys.stderr,
                )
                return 4
        else:
            print(
                f"Hinweis: keine Transkripte für diesen Lauf vorhanden -> "
                f"vorhandene Aggregat-Datei '{output}' bleibt unverändert."
            )

    # 6. Summary ausgeben ------------------------------------------------
    existing = sum(1 for _mp3, t in all_items if t.exists())
    print("\n" + "=" * 50)
    print("ZUSAMMENFASSUNG")
    print("=" * 50)
    print(f"  Verzeichnis      : {root}")
    print(
        f"  Modell           : {args.model_size} @ {device.upper()} "
        f"({compute_type})"
    )
    print(f"  Dateien gesamt   : {total}")
    print(f"  Neu transkribiert: {existing - skipped}")
    print(f"  Übersprungen     : {skipped}")
    print(f"  Fehlgeschlagen   : {len(failed)}")
    if work_total > 0:
        speed = audio_total / work_total
        print(
            f"  Audio verarbeitet: {audio_total / 60:.1f} min in "
            f"{work_total:.1f} s Rechenzeit"
        )
        print(f"  Durchsatz        : {speed:.1f}x Echtzeit")
        print(f"  Typische Sprache : {', '.join(f'{k} ({v}x)' for k, v in languages.items())}")
    print(f"  Timecodes        : {'an' if use_timestamps else 'aus'}")
    if args.enable_diarization:
        done_now_count = existing - skipped
        avg_speakers = (speaker_total / diarized_files) if diarized_files else 0.0
        print(
            f"  Diarization      : {diarized_files}/{done_now_count} Dateien "
            f"mit Sprechern (Ø {avg_speakers:.1f} pro Datei)"
        )
        if diarization_errors:
            print(
                f"  Diarization-Fehler: {diarization_errors} Datei(en) "
                "ohne Sprecher-Labels (siehe Warnungen oben)"
            )
    print(f"  Transkript-Ordner: {transcript_dir.resolve()} ({existing} Datei(en))")
    if not args.no_aggregate:
        print(f"  Aggregat-Datei   : {output.resolve()}")

    if failed:
        print("\nFehlgeschlagene Dateien (beim nächsten Lauf erneut versucht):")
        for path, error in failed:
            print(f"  - {path}: {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
