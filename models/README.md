# Local weights

The ImageNet checkpoints used for initialization belong in
`models/pretrained/` and are not versioned:

- `mobilenet_v3_small-047dcff4.pth`
- `resnet18-f37072fd.pth`

The pipeline requires known filenames, sizes, and hashes and never downloads
weights automatically. Checkpoints produced during training belong under
`artifacts/campaigns/` or `artifacts/runs/`, not in this directory.
