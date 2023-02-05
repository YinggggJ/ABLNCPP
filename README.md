# ABLNCPP

A Non-coding RNA Coding Potential Prediction Model Based on BiLSTM and Attention Mechanism

## Requirements

python == 3.7 <br>

pandas == 1.3.5 <br>

torch == 1.7.1 <br>

scipy == 1.5.4 <br>

scikit-learn == 1.0.2

## Dataset

The dataset of ABLNCPP is displayed in the data folder. 

high-confidence.fa includes the transcripts in the high-confident subset of LNCipedia 5. SPENCER.fa and LNCipedia5.fa are positive and negative samples of ABLNCPP, respectively.

>high-confidence.fa
>
>SPENCER.fa
>
>LNCipedia5.fa

## Usage

1. Data Preprocessing

preprocessing.py    - Preprocessing transcript information from fasta file and calculating corresponding features.

'$python preprocessing.py --withcp data/SPENCER.fa --withoutcp data/LNCipedia5.fa '

## Features
