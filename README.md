# EMG push-up assistant (AI in sports)

An assistant that watches an EMG envelope stream, counts push-up reps, estimates
fatigue (tempo loss + EMG amplitude rise) and coaches the athlete to end sets
near ~20 % tempo loss (reps in reserve) instead of training to failure.
**Inference uses EMG only. No video is used at inference.**

## Data provenance (state this in the report and demo)
- `recordings/pushup_set1_*`: real take. Rep timing measured from video
  (`reps_from_video.py`, 32 reps, ~60 ms resolution). The real EMG channel stayed
  at ~11 during the push-ups, so it carries no usable activation signal.
- `data/sim_sets/*.csv`: **SIMULATED EMG** (each has a `.json` with `"simulated": true`).
  set1 uses the real measured rep timing; sets 2-4 use generated timing.
- The assistant prints the data source in its banner every run.

## Files
- `record.py`, `live.py`, `arduino/`: sensor + camera recording tools
- `reps_from_video.py`: rep timing from video motion (optical flow, no pose model)
- `make_sets.py`: builds the simulated sets
- `emg_features.py`: EMG-only streaming rep tracker + fatigue metrics
- `knowledge/*.md` + `rag.py`: sports-science notes + BM25 retriever
- `assistant.py`: real-time assistant (RAG + DeepSeek)

## Run the assistant
Put your key in `.env` next to `assistant.py` (never paste it anywhere else):
```
DEEPSEEK_API_KEY=your_key_here
```
Replay a set in real time (use `--speed 5` for faster):
```
.venv/bin/python assistant.py --replay data/sim_sets/set1_real_timing.csv
```
Without a key it falls back to rule-based messages. Live sensor: `--live`.

## Demo window (video + EMG + metrics + assistant messages)
```
.venv/bin/python demo.py recordings/pushup_set1_s01_20261116_160217 --emg data/sim_sets/set1_real_timing.csv
```
Space = pause, q = quit. `--save demo.mp4` also writes a video (use as a backup),
`--show-measured` adds a strip with the real sensor reading from the take.
The video is display only; the assistant sees the EMG samples only.

## Sources behind the knowledge notes (verify before citing)
- Zourdos et al. (2016), RIR-based RPE scale, J Strength Cond Res
- Sanchez-Medina & Gonzalez-Badillo (2011), velocity loss and fatigue, Med Sci Sports Exerc
- Pareja-Blanco et al. (2017), 20 % vs 40 % velocity loss, Scand J Med Sci Sports
- De Luca (1997), surface EMG and fatigue, J Appl Biomech
