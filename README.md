# RL-ADM-OSC System

Dieses Projekt trainiert einen RL-Agenten mit manuellen Rewards über OSC.
Die Umgebung ist bewusst human-in-the-loop: Der nächste Schritt erfolgt erst,
wenn Reward oder ein Episode-Event gesendet wurde.

## Aktuelles Lernschema

- Observation: `actor_xyz + media_xyz` (6 Werte)
- Action: 3D Steuervektor (`/adm/obj/1/xyz`)
- Reward: ausschließlich manuell über `/reward`
- Episode-Ende:
  - `terminated=True` bei `/episode/end`
  - `truncated=True` bei `/episode/reset_manual`, `max_steps` oder `/training/stop`

## Manuelles Reward-Protokoll

Empfohlene konsistente Skala:

- `+1.0`: sehr gut
- `+0.5`: gut
- `0.0`: neutral
- `-0.5`: schlecht
- `-1.0`: stark falsch

## OSC-Adressen

- Zustand vom Simulator: `/adm/obj/101/xyz`
- Aktion vom Agenten: `/adm/obj/1/xyz`
- Reward: `/reward`
- Episode manuell resetten: `/episode/reset_manual`
- Episode manuell beenden: `/episode/end`
- Episode-Startzustand vom Env: `/episode/reset`
- Training stoppen: `/training/stop`

## Prinzip (Kamera → Agent → Mediengerät)

1. Kamera liefert die Actor-Position über `/adm/obj/101/xyz`.
2. Agent beobachtet `actor_xyz` und erzeugt daraus eine Aktion `/adm/obj/1/xyz`.
3. Mediengerät erhält das Signal über den Hub.
4. Mensch gibt Reward (`-2..+2`) basierend auf dem Ergebnis.
5. Agent passt die Strategie im Training an.

Wichtig: Der **nächste Zustand** ist primär die **neue Actor-Position** (Kamera), nicht nur die Mediengerät-Koordinate.
Die Mediengerät-Koordinate ist als Kontext im Zustand enthalten (`media_xyz`), damit der Agent weiß, was zuletzt gesendet wurde.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Startreihenfolge

In 4 Terminals:

```bash
python osc/hub.py
python osc/state_simulator.py
python osc/reward_input.py
python train/train.py
```

Optionale visuelle 3D-Darstellung:

```bash
python osc/visualizer.py
```

Hinweis:
- Der Akteur wird aus `/adm/obj/101/xyz` gezeichnet.
- Das Scheinwerfer-Ziel wird aus `/adm/obj/1/xyz` gezeichnet.
- Mit `--axes xz` kannst du alternativ die `x/z`-Projektion anzeigen.

Hinweis: `train/train.py` verwendet aktuell feste Defaults (z. B. `max_steps=100`).
Wenn du andere Werte willst, passe die Parameter im `train()`-Aufruf an.

## Modell speichern und wieder laden (.zip)

Das Training verwendet `train/models/final_model.zip` als Standard.
Wenn die Datei existiert, wird sie geladen und das Training fortgesetzt.

Speichern erfolgt automatisch:
- **nach jeder Episode** bei `/episode/end`, `/episode/reset_manual` oder `max_steps`
- **bei Trainingsende** (z. B. Timesteps erreicht)
- **bei `/training/stop`** (Taste `q` in `osc/reward_input.py` beendet das Training sauber)

Hinweis: Damit der finale Save im `finally`-Block ausgeführt wird, das Training sauber beenden
und den Prozess nicht hart stoppen (z. B. IDE-Stop/Terminal schließen).

## Episoden-Zusammenfassung

Nach einer Episode (siehe oben) wird eine Summary in der Konsole ausgegeben, inkl.:
- Episodenlänge (Steps)
- diskontierter Return und kumulativer Reward
- Max-/Mean-Reward
- Policy-Statistiken (Mean/Std/Entropy)
- Gesamt-Timesteps

Diese Ausgabe kommt aus `EpisodeSummaryCallback` in `train/train.py`.
