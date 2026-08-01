# Remap FER2013 labels to a stable output order

Every classifier artifact uses `0=angry`, `1=happy`, `2=neutral`, and `3=sad`.
The CSV loader explicitly maps original FER2013 IDs `0→0`, `3→1`, `6→2`, and
`4→3`, while original IDs 1 (disgust), 2 (fear), and 5 (surprise) are excluded.
Both export formats and every metrics document publish the same lowercase label
list. This preserves the repository's existing output contract and prevents raw
CSV ordering from silently changing API or microcontroller interpretations.
