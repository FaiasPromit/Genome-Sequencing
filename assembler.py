import subprocess
import os
from typing import Optional

import pysam

from utils import get_logger
import config

logger = get_logger(__name__)

def run_spades(fastq_path: str, output_dir: str) -> Optional[str]:
    """
    Runs SPAdes assembler on a given FASTQ file.
    
    Returns: Path to the assembled contigs FASTA file or None if assembly fails.
    """
    spades_command = [
        "spades.py",
        "-s", fastq_path,
        "-o", output_dir,
        "--only-assembler",
        "-t", "8", # Use 8 threads, adjust as needed
        "-m", "32" # Use 32 GB RAM, adjust as needed
    ]
    
    try:
        logger.info(f"Running SPAdes: {' '.join(spades_command)}")
        subprocess.run(spades_command, check=True, capture_output=True, text=True)
        
        contigs_path = os.path.join(output_dir, "contigs.fasta")
        if os.path.exists(contigs_path):
            logger.info(f"SPAdes assembly successful. Contigs at {contigs_path}")
            return contigs_path
        else:
            logger.warning("SPAdes completed but contigs.fasta not found.")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"SPAdes failed with exit code {e.returncode}")
        logger.error(f"SPAdes STDERR: {e.stderr}")
        return None


def assemble_region(bam_path: str, chrom: str, start: int, end: int, variant_id: str) -> Optional[str]:
    """
    Extracts reads from a genomic region, writes them to a FASTQ,
    and performs local de novo assembly.

    Returns: The sequence of the longest assembled contig.
    """
    if not os.path.exists(config.ASSEMBLY_DIR):
        os.makedirs(config.ASSEMBLY_DIR)

    fastq_path = os.path.join(config.ASSEMBLY_DIR, f"{variant_id}.fastq")
    assembly_output_dir = os.path.join(config.ASSEMBLY_DIR, variant_id)
    
    try:
        # 1. Extract reads from the region using pysam
        logger.info(f"Extracting reads for {variant_id} in region {chrom}:{start}-{end}")
        with pysam.AlignmentFile(bam_path, "rb") as bamfile, open(fastq_path, "w") as fq:
            count = 0
            for read in bamfile.fetch(chrom, start, end):
                if not read.is_unmapped and read.query_sequence:
                    fq.write(f"@{read.query_name}\n")
                    fq.write(f"{read.query_sequence}\n")
                    fq.write("+\n")
                    fq.write(f"{pysam.qualities_to_qualitystring(read.query_qualities)}\n")
                    count += 1
            logger.info(f"Extracted {count} reads.")
            if count < 5: # Not enough reads to assemble
                logger.warning("Fewer than 5 reads found, skipping assembly.")
                return None

        # 2. Run SPAdes
        contigs_path = run_spades(fastq_path, assembly_output_dir)

        if not contigs_path:
            return None

        # 3. Get the longest contig from the assembly
        longest_contig_seq = ""
        with open(contigs_path, "r") as f:
            from Bio import SeqIO
            longest_len = 0
            for record in SeqIO.parse(f, "fasta"):
                if len(record.seq) > longest_len:
                    longest_len = len(record.seq)
                    longest_contig_seq = str(record.seq)
        
        logger.info(f"Assembly for {variant_id} resulted in longest contig of size {len(longest_contig_seq)}")
        return longest_contig_seq

    finally:
        # Clean up temporary files
        if os.path.exists(fastq_path):
            os.remove(fastq_path)
        # You might want to keep the assembly_output_dir for debugging
        # import shutil; shutil.rmtree(assembly_output_dir)
    
    return None