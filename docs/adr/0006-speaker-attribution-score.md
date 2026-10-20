# ADR-006: Speaker Attribution Score

## Status

Accepted

## Context

ADR 0001 of stt-bridge (`docs/adr/0001-speaker-attribution.md` in that repository) adds speaker attribution to the application. Its section 6 gives the measures of that change. It also names the command `modebench speakers score`. The speech suite of this repository measures one speaker accuracy for each replayed audio file (`modebench.stt.speakers`). It does not measure a full meeting with two channels, and it does not measure names.

A meeting of stt-bridge has two channels: the microphone (the operator) and the loopback (the persons on the call). Without headphones, the microphone also records the loudspeaker. The application then has the same words on the two channels. ADR 0001 calls this an echo.

## Decision

1. Command:
   `modebench speakers score --gold <index> --hypothesis <detail> --mode <off|native|fusion>` scores one meeting. `--meeting` selects the meeting of the index. The default is the file name of the hypothesis without its extension. The score goes to `runs/speakers/<mode>/<meeting>.score.json`, or to `--out`. Git ignores `runs/`, because a score can contain names of persons.

2. Gold Set:
   The index is a YAML file. Each meeting has an RTTM file (who spoke when), an optional UEM file (the scored parts), an optional SegLST file (the words, in the JSON form of MeetEval), the real names of its speakers, and an optional time offset. The paths are relative to the index. Keep the gold set below `data/private/`, because it contains the words and the names of real persons.
   - An RTTM label is unique in its file and tells the channel (`mic-op`, `sys-ana`). The score does not read the channel field.
   - Without a UEM file, the scored part is from the start of the first turn to the end of the last turn. The UEM is always explicit for pyannote.metrics.
   - A name can have more than one correct form, for example `[Ana Souza, Ana]`.
   - `offset_s` is added to the times of the application. Use it when the gold audio does not start at the start of the meeting.

3. Hypothesis:
   The hypothesis is the detail of one meeting from `GET /api/meetings/{id}/detail`, or the `.json` export of the page, which has the same detail below `meeting`.
   - For `native` and `fusion`, the speaker of a line is its meeting label (`speaker_label`). One label is one person on the two channels.
   - For `off`, a label of the speech service is correct only on its channel. The speaker is the channel and the label (`mic:A`, `system:A`). A line without a label is `<channel>:?`.
   - `native` or `fusion` with no meeting label stops with an error, because that meeting was recorded with attribution off.

4. Measures:

   | Measure | Tool | Setting |
   | --- | --- | --- |
   | DER, JER | pyannote.metrics 4.2.0 | `collar=0.0`, `skip_overlap=False`, explicit UEM, each annotation reduced to its support |
   | Loopback DER | pyannote.metrics 4.2.0 | The same, with the `sys-` turns and the loopback lines only |
   | Operator confusion | Own code | Seconds of microphone speech under a speaker of the wrong side, over the microphone speech that meets a `mic-` or `sys-` turn |
   | cpWER, tcpWER | MeetEval 0.4.3 | tcpWER collar 5 s, `normalizer=None` |
   | WDER and WER of the same alignment | Own code, word alignment of jiwer | (S_IS + C_IS) / (S + C) |
   | Name recall and precision | Own code | Only the names the application found (origin `rule` or `model`), match of speakers by shared time, names without accents and letter case |
   | Echo rate and marked rate | Own code | Tolerance of 1 s |

   - The words go to MeetEval and to the WDER after the Portuguese normaliser of this repository (`modebench.stt.normalize`). MeetEval gets no normaliser of its own, because its normalisers remove accented letters.
   - The WDER, cpWER and tcpWER do not read an echo line, because the line of the call has the same words.
   - DER and JER read an echo line. An echo with the wrong speaker is speech of the wrong speaker at that time. An echo with the correct speaker adds nothing, because the support of the annotation merges it with the line of the call.
   - The side of a reference speaker is the start of its label: `mic-` is the operator, `sys-` is a person of the call. The operator confusion counts the operator under a person of the call, and a person of the call (an echo) under the operator. Without attribution, each microphone line is the operator. Speech of two persons at the same time counts against the measure.
   - The match of speakers is the Hungarian algorithm on a table of agreement: shared words for the WDER, shared seconds for the names. A speaker with no agreement has no match.
   - A name from the configuration or from a person is not a name that the application found. The score leaves it out of the recall and of the precision.
   - Name recall is the share of the named reference speakers whose match has a correct found name. Name precision is the share of the found names that are correct. A found name is judged only when the gold set names its reference speaker.
   - The score lists each name of the application with its origin, its confidence, its reference speaker and its verdict. It also gives the threshold of the application (75 percent), so a change of a confidence is a measured change.
   - The echo rate is the share of the microphone words that a line of the call has at the same time, with 1 s of tolerance. It compares bags of words, so a common word can count. It is an upper limit. The marked rate is the share of the microphone words in the lines that the application marked as an echo.

5. Optional Libraries:
   pyannote.metrics and MeetEval are not dependencies of modebench. Only this command uses them. MeetEval 0.4.3 is a source package only, so its install compiles C++ code and needs a compiler. Add the two libraries for one command with `uv run --with`. Without them, the score does not measure DER, JER, cpWER and tcpWER, and its `not_measured` list tells which library to add.

6. Latency and Cost:
   - With `STT_BRIDGE_LATENCY_PROBE` in its environment, the recorder of stt-bridge writes a probe line when a settled line arrives, when its speaker is recorded, and when a speaker gets a name. A probe line carries labels, never names.
   - `--probe-log` reads the error stream of the recording. The added latency of attribution is the time from the arrival of a line to the record of its speaker. The name delay of a line is the time from its attribution to the first name of its speaker, for each line said before its speaker had a name. The score gives p50 and p95 of both.
   - `--cost-usd` takes the total of `stt-bridge billing --meeting`. The cost per hour divides it by the length of the meeting, from its start and its end.

7. Public Sets:
   - `modebench speakers import-rttm` copies the RTTM and the UEM of one meeting of a public set (AMI, VoxConverse) into the gold set. It puts the channel prefix on each label (`sys-` for a meeting played as the computer audio) and adds the meeting to the index.
   - stt-bridge plays the audio of the meeting through the whole pipeline with `record --system-file`, and writes the detail with `export --json`. The loudspeakers do not play anything.

## Consequences

- One command gives the numbers that ADR 0001 needs for its acceptance:
  - the DER on the loopback;
  - the confusion of the operator with a person of the call on the microphone;
  - the name accuracy, with the confidences and the threshold of the application.
- The type check and the tests of this repository run without the two optional libraries. Two tests use them only when they are installed. The adapter (`modebench/speakers/external.py`) is not in the coverage threshold, as the other adapters that need a resource outside the tests.
- A line without time (Gemini sends `0..0`) has no duration. DER and JER do not count it. The WDER puts it at its start time, so its words can be out of order.
- The time offset must be known. A wrong offset makes DER and the name match wrong, but it does not change the WDER, because the WDER uses the order of the words.

## Commands

```sh
make check
modebench speakers import-rttm --rttm ES2004a.rttm --uem ES2004a.uem \
  --meeting ES2004a --prefix sys- --out-dir data/private/speakers
uv run --with pyannote.metrics==4.2.0 --with meeteval==0.4.3 \
  modebench speakers score --gold data/private/speakers/gold.yaml \
  --hypothesis ES2004a.json --mode fusion --probe-log ES2004a.err --cost-usd 0.13
```

## References

- stt-bridge ADR 0001, section 6: the measures, the gold set rules and the acceptance.
- Shafey et al., Interspeech 2019, the WDER: https://doi.org/10.21437/Interspeech.2019-1943
- MeetEval and the tcpWER collar of 5 s: https://arxiv.org/abs/2307.11394
- MeetEval 0.4.3 on PyPI (source package only): https://pypi.org/project/meeteval/0.4.3/
- pyannote.metrics 4.2.0 on PyPI (Python 3.11 or newer): https://pypi.org/project/pyannote.metrics/4.2.0/
- `cpwer` and `tcpwer` signatures: https://github.com/fgnt/meeteval/blob/v0.4.3/meeteval/wer/api.py
- `DiarizationErrorRate` and `JaccardErrorRate`: https://github.com/pyannote/pyannote-metrics/blob/4.2.0/src/pyannote/metrics/diarization.py
