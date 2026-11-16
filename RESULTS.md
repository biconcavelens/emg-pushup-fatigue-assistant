# Results

## 1. What was measured and what was simulated

| Item | Origin |
|---|---|
| Video of one push-up set (32 reps, about 63 s, 16.7 fps) | Recorded, one subject |
| Rep timing (bottom of each rep) | **Measured** from the video by optical-flow motion analysis (no pose model); resolution about 60 ms, one camera frame |
| EMG sensor reading during the set | **Measured** with the Advancer Muscle Sensor v3 + Arduino Uno; it stayed at about 10.5 ADC counts (rest level) throughout, so it contains no usable activation signal |
| EMG envelope used by the assistant and shown in the demo | **Simulated** from the measured rep timing (set 1) or from generated timing (sets 2 to 4) |

All EMG in the demo and in the evaluation below is simulated. The amplitude rise
in the simulated sets was put in by the simulator (a fatigue gain), so it is an
assumption of the simulation, not a physiological finding.

## 2. Measured fatigue pattern in the real set (from video)

Over the 32 reps the time between consecutive reps grew from a median of 1.22 s
(first three intervals) to 3.88 s (last three), with the final interval reaching
4.97 s. This is a tempo loss of about 69 % between the start and the end of the
set, consistent with a set taken to and beyond failure. The tempo slowed
gradually: intervals stayed near 1.1 to 1.4 s for roughly the first 14 reps, then
lengthened steadily.

## 3. EMG-only rep tracker and fatigue status

The tracker uses only the EMG envelope (no video). Applied to the four simulated
sets it counted every rep:

| Set | Simulated how | True reps | Detected | Status first reached at rep: building fatigue / stop zone / at failure | Final tempo loss |
|---|---|---|---|---|---|
| 1 | real timing, to failure | 32 | 32 | 6 / 16 / 21 | 69.9 % |
| 2 | generated, to failure | 28 | 28 | 10 / 12 / 17 | 66.1 % |
| 3 | generated, fatigued | 22 | 22 | 7 / 10 / 14 | 63.7 % |
| 4 | generated, stopped near 20 % | 16 | 16 | 12 / never / never | 16.5 % |

Caveat: the simulated EMG is built from the same rep timing the tracker is
scored against, so this table tests the tracker and the status rules, not
real-muscle behaviour. Counting accuracy on real push-up EMG is not tested.

## 4. Stopping recommendation

On the real-timing set the tempo-loss rule (stop zone at 20 % loss) triggers at
rep 16 of 32, about halfway through the set that was taken to failure. Under the
assistant's guidance (end the set with reps in reserve, then rest 2 to 3
minutes), the athlete would have stopped after roughly half the reps and avoided
the grinding second half, where each rep took 2 to 5 times longer. Set 4 shows
the intended outcome: a 16-rep set that ends at 16.5 % tempo loss, below the stop
zone.

## 5. Assistant (RAG + LLM)

The assistant retrieves paragraphs from seven short notes (reps in reserve,
velocity loss, EMG and fatigue, rest and recovery, push-up technique, reading the
EMG stream, set planning) with a BM25 retriever and passes them with the current
metrics to the DeepSeek chat model, which returns one or two sentences. In the
demo it gave messages such as ending the set when tempo loss and amplitude rise
were high and resting 2 to 3 minutes. Message quality was checked by reading
examples; no systematic evaluation was done.

## 6. Limitations

- One subject, one set of real data.
- The EMG sensor produced no usable signal during push-ups (no gain control on
  the board; electrode placement and spacing were not resolved), so activation
  behaviour is simulated.
- The sensor outputs an envelope, so frequency-based fatigue measures cannot be
  computed.
- Tempo loss is a proxy for velocity loss (rep duration at fixed range of motion).
- The knowledge notes are short summaries written for this project; check the
  cited papers (Zourdos 2016, Sanchez-Medina 2011, Pareja-Blanco 2017, De Luca
  1997) before quoting numbers.
- The assistant gives general training guidance, not medical advice.

## 7. Future work

Record several sets and subjects with a working EMG channel (gain control, fixed
electrode spacing), replace the simulated envelope with real recordings, and test
the tracker and stop rule on them, including held-out sets.
