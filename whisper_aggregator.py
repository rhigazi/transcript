#!/usr/bin/env python3
"""Whisper-Aggregator-CLI

Scannt ein Verzeichnis rekursiv nach MP3-Dateien, transkribiert jede Datei
mit faster-whisper und schreibt:

  1. je MP3 eine EINZELNE Textdatei in das Transcript-Verzeichnis
     (Verzeichnisstruktur 1:1 gespiegelt) -> RESUME-FÄHIG
  2. optional alle Transkripte aggregiert in eine globale Textdatei

Resume-Logik: Vorhandene Transcript-Dateien werden übersprungen. Durch
atomares Schreiben (Schreiben auf .part + rename) kann ein abgebrochener
Lauf jederzeit neu gestartet werden und fährt an der letzten Stelle fort.

Aufruf-Beispiel:
    python whisper_aggregator.py --path ./podcasts --transcript-dir ./transcript
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # tqdm ist optional -> Fallback-Wrapper
    tqdm = None  # type: ignore[assignment]

# Erlaubte Whisper-Modellgrößen (PRD: tiny, base, small, medium, large)
MODEL_SIZES = ("tiny", "base", "small", "medium", "large", "turbo")
DEVICES = ("cpu", "cuda")
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
        default="base",
        help="Whisper-Modellgröße (Turbo = schnellster Modus).",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="cpu",
        help="Ausführungsgerät: CPU oder GPU (CUDA).",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Override für den Rechentyp (z.B. int8, float16, float32). "
        "Standard: int8 (CPU) bzw. float16 (CUDA).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optionale Sprachangabe (z.B. 'de', 'en'). Standard: Auto-Erkennung.",
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
def load_model(model_size: str, device: str, compute_type: str):
    """Initialisiert das faster-whisper Modell (CPU/GPU)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "FEHLER: 'faster-whisper' ist nicht installiert.\n"
            "  Installation: pip install faster-whisper"
        ) from exc

    print(
        f"Lade Whisper-Modell '{model_size}' auf {device.upper()} "
        f"(compute_type={compute_type}) ..."
    )
    return WhisperModel(
        model_size,
        device=device,
        device_index=0,
        compute_type=compute_type,
    )


def transcribe_file(model, file_path: Path, language: Optional[str]) -> str:
    """Transkribiert eine einzelne Audiodatei und gibt den Text zurück."""
    segments, _info = model.transcribe(
        str(file_path),
        language=language,
        vad_filter=True,  # Rausch-/Pausenfilter -> saubererer Turbo-Flow
    )
    return "\n".join(segment.text.strip() for segment in segments).strip()


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

    return _Plain(files)


# --------------------------------------------------------------------------
# Main Workflow
# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # 1. CLI Arguments parsen -------------------------------------------
    root: Path = args.path.expanduser().resolve()
    transcript_dir: Path = args.transcript_dir.expanduser()
    output: Path = args.output.expanduser()
    compute_type: str = args.compute_type or COMPUTE_TYPES[args.device]

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
    else:
        # 3. Whisper Modell initialisieren ------------------------------
        try:
            model = load_model(args.model_size, args.device, compute_type)
        except SystemExit:
            raise
        except Exception as exc:  # z.B. fehlende CUDA-Laufzeit
            print(f"FEHLER beim Laden des Modells: {exc}", file=sys.stderr)
            return 3

        # 4. Transkribieren -> EINZELDATEIEN (atomar geschrieben) -------
        skipped = len(already_done)
        failed = []
        done_now = 0
        bar = progress_bar(todo)

        for mp3, target in bar:
            try:
                text_content = transcribe_file(model, mp3, args.language)
                write_text_atomic(target, text_content)
                done_now += 1
                if tqdm is not None:
                    bar.set_postfix_str(
                        f"ok={done_now} Fehler={len(failed)} "
                        f"überspr.{skipped}"
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
        f"  Modell           : {args.model_size} @ {args.device} "
        f"({compute_type})"
    )
    print(f"  Dateien gesamt   : {total}")
    print(f"  Neu transkribiert: {existing - skipped}")
    print(f"  Übersprungen     : {skipped}")
    print(f"  Fehlgeschlagen   : {len(failed)}")
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
