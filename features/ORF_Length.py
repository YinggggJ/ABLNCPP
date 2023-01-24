import sys
sys.path.append("..")
from features import ORF


def extract_feature_from_seq(seq, stt, stp):
    """extract features of sequence from fasta entry"""

    stt_coden = stt.strip().split(",")
    stp_coden = stp.strip().split(",")
    RNA_seq = seq.upper()
    RNA_size = len(seq)
    tmp = ORF.ExtractORF(RNA_seq)
    (CDS_size1, CDS_integrity, CDS_seq1) = tmp.longest_ORF(start=stt_coden, stop=stp_coden)
    return RNA_size, CDS_size1, CDS_integrity


start_codons = "ATG"
stop_codons = "TAG,TAA,TGA"
Coverage = 0


def len_cov(seq):
    RNA_size, CDS_size, CDS_integrity = extract_feature_from_seq(seq=seq, stt=start_codons, stp=stop_codons)
    RNA_len = RNA_size
    CDS_len = CDS_size
    Coverage = float(CDS_len) / RNA_len
    Integrity = CDS_integrity
    return CDS_len, Coverage, Integrity
