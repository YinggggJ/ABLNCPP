# ABLNCPP
A Non-coding RNA Coding Potential Prediction Model Based on BiLSTM and Attention Mechanism
## Requirements
python == 3.7 <br> <br>
pandas == 1.3.5 <br> <br>
torch == 1.7.1 <br> <br>
scipy == 1.5.4 <br> <br>
scikit-learn == 1.0.2
## Dataset
The dataset of ABLNCPP is displayed in the data folder. 

high-confidence.fa includes the transcripts in the high-confident subset of LNCipedia 5. SPENCER.fa and LNCipedia5.fa contain ncRNA sequences with coding potential and ncRNA sequences without coding potential, respectively. 

>high-confidence.fa

>SPENCER.fa

>LNCipedia5.fa
## Usage
1. Data Preprocessing
preprocessing.py    -Obtaining the sequence information from the fasta file and calculate the corresponding features
## Features
