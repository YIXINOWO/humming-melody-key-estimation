# Third-party resources

No third-party dataset, repository snapshot, or model weight is redistributed
in this release. Obtain each resource from its upstream provider and follow its
licence or terms of use.

| Resource | Upstream route | Used for | Included here |
|---|---|---|---|
| MTG-QBH/Molina corpus | Original source cited in Molina et al. (2014, 2015) | Main 38-recording evaluation and 132-recording description | No audio or annotations |
| HumTrans | https://huggingface.co/datasets/dadinghh2/HumTrans | 765-recording validation and 769-recording test | No audio, reference MIDI, or upstream baseline archives |
| ROSVOT | https://github.com/RickyL-2000/ROSVOT | Identical-waveform neural baseline | No source snapshot or weights |
| ROSVOT checkpoints | Upstream link: https://drive.google.com/file/d/1JNtNT37KiLq9uFQqHk7JFs-3trxd3bRh/view | ROSVOT, RWBD, and RMVPE inference | No checkpoint archive or extracted weights |
| torchcrepe / CREPE | Upstream Python package and model download | F0 and confidence extraction | No cached model files |
| VOCANO, SheetSage, MIR-ST500, JDC-STP | Links distributed by the HumTrans evaluation project | External baseline comparison | No upstream code or MIDI archives |

Verified ROSVOT checkpoint archive SHA-256:

```text
b6055e81315b93415c9bd7fc48e10a28a3da1bea960cab7385483bd7443ba852
```

Verified extracted checkpoint hashes:

```text
checkpoints/rmvpe/model.pt   19dc1809cf4cdb0a18db93441816bc327e14e5644b72eeaae5220560c6736fe2
checkpoints/rosvot/config.yaml 2ad2cb756623418c471b7dc2f56175cce88b69a70b4a2c354fa1a78525aa54e2
checkpoints/rosvot/model.pt  7501fb5f913d971c2f51bcb3063b930027b03206581820a4d2bfdc394c9c3fcb
checkpoints/rwbd/config.yaml 3bb41f1d9eaa85aa1b3e5b6d94fff4ab4affb39719028ab69b4505974b9a1bc7
checkpoints/rwbd/model.pt    0bc2d42a6d4b7a05436deb937e2deda1c12de49e5687cfda0bdf6a430120dcd2
```

These hashes identify the files used for the manuscript experiments; they do
not grant redistribution rights.

