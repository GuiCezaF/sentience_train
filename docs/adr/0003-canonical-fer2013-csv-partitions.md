# Use the original FER2013 CSV and preserve its partitions

The canonical dataset source is the original `fer2013.csv`. After retaining
only Angry, Happy, Neutral, and Sad, rows marked `Training` train the model,
`PublicTest` drives model selection and early stopping, and `PrivateTest`
remains untouched until final evaluation. This replaces the repository's
current `ImageFolder` assumption and avoids creating a validation set from, or
tuning checkpoints against, the final test data.
