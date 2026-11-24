# Original research notes

The original project evaluated whether correcting simultaneous EEG/fMRI recordings preserves the EEG features needed for a downstream fMRI encoding model. The EEG benchmark and the voxelwise analysis were different stages of the project.

## Artifact benchmark

The reported comparison used four correction strategies on a 20-participant simultaneous EEG/fMRI dataset. Two hybrid approaches removed 96.2% and 96.9% of heartbeat artifact and also distorted alpha activity. I built a ridge regression evaluation with five contiguous test blocks, a three-volume exclusion buffer, and 200 circular time shifts for the null comparison.

## Why the voxelwise analysis stopped

The scan alignment check found 125 scan intervals for 120 stored volumes and six plausible timing alignments. The metadata could not resolve the alternatives. An alpha-focused recovery test also missed its preset criteria: rank agreement was 0.67 against a threshold of 0.80, and spatial agreement was 0.23 against 0.70. I held the voxelwise result and specified the trigger records, timing metadata, and recovery checks to require on a new dataset.

These are historical project results reported by Devarshi Dalal. The public walkthrough exercises similar validation decisions with a synthetic series. The project used the [Simultaneous EEG-fMRI dataset, version 2](https://doi.org/10.17632/crhybxpdy6.2) from Jonathan Gallego-Rudolf, Maria Corsi Cabrera, Luis Concha, Josefina Ricardo Garcell, and Erick H. Pasaye Alcaraz. Their [2022 study](https://doi.org/10.3389/fnins.2022.951321) describes the acquisition and EEG spectral preservation experiments.
