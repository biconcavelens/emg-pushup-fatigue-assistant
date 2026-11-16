# How this assistant reads the EMG envelope stream

Input: one envelope sample stream (ADC counts, 0 to 1023, about 500 Hz). No camera is used at inference.

Rep detection: the envelope is smoothed and each rep appears as a peak of muscle activation. A rep is confirmed when the envelope falls to about 60 % of a peak that is large enough compared with recent peaks. Rep time is the time between consecutive peaks, in milliseconds.

Metrics per set: rep count; tempo loss (velocity-loss proxy) from the interval between reps versus the first reps; EMG amplitude change versus the first three reps; status label (fresh, building fatigue, stop zone, at failure).

Interpretation guide: tempo loss below 10 % with stable amplitude means fresh. Tempo loss of 10 to 20 % with rising amplitude means fatigue is building but the set is still productive. Tempo loss of 20 to 35 % is the stop zone: recommend ending the set with reps in reserve. Tempo loss above 35 %, or reps taking more than twice as long as at the start, means at or near failure.

After the set ends (no reps for several seconds): summarise reps, tempo loss, amplitude rise, and suggest a rest time based on how close to failure the set got.

Caveats: an envelope sensor cannot show frequency-based fatigue; placement and gain change the amplitude scale; one subject. Advice is general training guidance, not medical advice.
