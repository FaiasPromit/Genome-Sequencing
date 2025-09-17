import pandas as pd
import pysam
import pyfaidx
from tqdm import tqdm
import os

import config
from utils import get_logger
from assembler import assemble_region

logger = get_logger(__name__)

def get_alt_sequence(fasta, chrom: str, pos: int, ref: str, alt: str, sv_type: str, sv_len: int, end: int) -> str:
    """Constructs the alternate sequence based on the SV type."""
    flank_size = config.FLANKING_SIZE
    
    try:
        # Get flanking sequences
        left_flank = fasta[chrom][pos - flank_size : pos].seq
        right_flank = fasta[chrom][end : end + flank_size].seq
        
        if sv_type == 'DEL':
            return left_flank + right_flank
        
        elif sv_type == 'INS':
            # For insertions, the 'alt' field in the VCF should contain the inserted sequence
            ins_seq = alt[len(ref):]
            return left_flank + ins_seq + right_flank
            
        elif sv_type == 'DUP':
            dup_seq = fasta[chrom][pos:end].seq
            return left_flank + dup_seq + dup_seq + right_flank
            
        elif sv_type == 'INV':
            inv_seq = fasta[chrom][pos:end].seq
            inversed_seq = str(pyfaidx.Sequence(inv_seq).reverse.complement)
            return left_flank + inversed_seq + right_flank
            
    except (pyfaidx.FetchError, ValueError) as e:
        logger.warning(f"Could not fetch sequence for {chrom}:{pos}-{end}. Error: {e}")
        return None
        
    return None


def main():
    logger.info("Starting preprocessing pipeline...")

    # Load reference genome
    logger.info(f"Loading reference genome from {config.REF_GENOME}")
    fasta = pyfaidx.Fasta(config.REF_GENOME)

    # Load GIAB truth set
    logger.info(f"Loading GIAB truth set from {config.TRUTH_VCF}")
    truth_vcf = pysam.VariantFile(config.TRUTH_VCF)
    
    truth_svs = []
    for record in truth_vcf.fetch():
        if "SVTYPE" in record.info and record.info["SVTYPE"] in ['DEL', 'INS', 'DUP', 'INV']:
            sv_len = record.info.get("SVLEN", [0])[0]
            truth_svs.append({
                "chrom": record.chrom,
                "start": record.start,
                "end": record.stop,
                "sv_type": record.info["SVTYPE"],
                "sv_len": abs(sv_len)
            })
    truth_df = pd.DataFrame(truth_svs)
    logger.info(f"Loaded {len(truth_df)} truth SVs.")

    processed_records = []
    
    # Iterate through candidate VCFs
    for caller, vcf_path in config.CANDIDATE_VCFS.items():
        if not os.path.exists(vcf_path):
            logger.warning(f"Candidate VCF not found, skipping: {vcf_path}")
            continue
            
        logger.info(f"Processing candidate VCF from {caller}: {vcf_path}")
        candidate_vcf = pysam.VariantFile(vcf_path)
        
        for record in tqdm(candidate_vcf.fetch(), desc=f"Processing {caller}"):
            try:
                sv_type = record.info.get("SVTYPE")
                if not sv_type or sv_type not in ['DEL', 'INS', 'DUP', 'INV', 'BND']:
                    continue

                sv_len_info = record.info.get("SVLEN")
                sv_len = abs(sv_len_info[0] if isinstance(sv_len_info, tuple) else sv_len_info)
                
                if not (config.MIN_SV_LENGTH <= sv_len <= config.MAX_SV_LENGTH):
                    continue

                chrom = record.chrom
                start = record.start
                end = record.stop
                
                # --- Assign Label by matching with truth set ---
                label = 0
                overlapping_truth_svs = truth_df[
                    (truth_df['chrom'] == chrom) &
                    (truth_df['sv_type'] == sv_type) &
                    (truth_df['start'] < end) &
                    (truth_df['end'] > start)
                ]

                if not overlapping_truth_svs.empty:
                    for _, truth_sv in overlapping_truth_svs.iterrows():
                        # Calculate reciprocal overlap
                        intersection_start = max(start, truth_sv['start'])
                        intersection_end = min(end, truth_sv['end'])
                        intersection_len = intersection_end - intersection_start
                        
                        union_len = (end - start) + (truth_sv['end'] - truth_sv['start']) - intersection_len
                        if union_len == 0: continue
                        
                        overlap_score = intersection_len / float(union_len) # Jaccard index
                        
                        len_similarity = min(sv_len, truth_sv['sv_len']) / max(sv_len, truth_sv['sv_len'])
                        
                        if overlap_score > config.RECIPROCAL_OVERLAP and len_similarity > config.RECIPROCAL_OVERLAP:
                            label = 1
                            break # Found a match
                
                # --- Construct Sequences ---
                ref_seq_region = fasta[chrom][start - config.FLANKING_SIZE : end + config.FLANKING_SIZE]
                ref_seq = ref_seq_region.seq

                if sv_type == 'BND': # Use assembler for complex breakends
                    alt_seq = assemble_region(config.BAM_FILE, chrom, start-100, start+100, record.id)
                else:
                    alt_seq = get_alt_sequence(fasta, chrom, start, record.ref, record.alts[0], sv_type, sv_len, end)

                if ref_seq and alt_seq:
                    processed_records.append({
                        "id": record.id if record.id else f"{caller}_{chrom}_{start}",
                        "ref_seq": ref_seq,
                        "alt_seq": alt_seq,
                        "sv_type": sv_type,
                        "label": label
                    })

            except (KeyError, TypeError, ValueError) as e:
                logger.debug(f"Skipping record {record.id if record.id else 'NO_ID'} due to error: {e}")
                continue

    # Save to file
    output_df = pd.DataFrame(processed_records)
    output_df.to_csv(config.PROCESSED_DATA_TSV, sep="\t", index=False)
    logger.info(f"Preprocessing complete. Saved {len(output_df)} records to {config.PROCESSED_DATA_TSV}")


if __name__ == "__main__":
    main()