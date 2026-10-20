"""The speaker attribution of the stt-bridge application, measured against a gold set.

The measure follows section 6 of ADR 0001 of stt-bridge
(`docs/adr/0001-speaker-attribution.md` in that repository) and ADR 0006 of
this repository. DER and JER come from pyannote.metrics, cpWER and tcpWER come
from MeetEval, and WDER, the name accuracy and the echo rate come from the code
of this package.
"""
