# Audio for the speech to text suite

The repository has no audio. The suite measures your own recordings.

## Add audio

1. Make the directory `data/private/stt/`. Git ignores it.
2. Put the WAV files there. Each file must be PCM, 16 bit. One channel is best.
   The suite mixes a file with two channels down to one channel.
3. Copy `manifest.example.yaml` to `data/private/stt/manifest.yaml`.
4. For each file, write the utterances: the start and the end in milliseconds,
   the speaker, and the words that were said.
5. In `configs/stt.toml`, set `private_data_ok = true` for each provider whose
   data policy permits this audio. A private manifest does not run without it.

A manifest is private if it says `visibility: private` or if it is below
`data/private/`. One of the two is sufficient, so a manifest in the private
directory stays private when its header says `public`.

## Rules for the reference text

- Write the words as they were said. Do not correct the grammar of the speaker.
- Numbers can be digits or words. The normalisation makes them equal.
- Punctuation and capital letters have no effect on the error rates.
- Give the same speaker the same label in all the utterances of a file.

## Try the suite with no audio

```
uv run modebench run --suite stt --profile smoke --dry-run
```

A dry run uses a fake server and a synthetic manifest. It sends nothing to a
network, and its numbers are not real.
