import torch
import pysam
import pyfaidx
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

import config
from utils import get_logger
from model import DNABERTSVClassifier
from preprocess import get_alt_sequence # Reuse logic

logger = get_logger(__name__)

def predict(model, tokenizer, device, ref_seq, alt_seq):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            ref_seq,
            alt_seq,
            return_tensors="pt",
            max_length=config.MAX_TOKEN_LENGTH,
            padding="max_length",
            truncation=True
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Return the probability of the 'true' class (class 1)
        confidence_score = probabilities[0, 1].item()
        return confidence_score

def main(args):
    logger.info("Starting prediction pipeline...")
    
    # 1. Load Model and Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = DNABERTSVClassifier(config.MODEL_NAME)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    logger.info(f"Model loaded from {args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # 2. Load reference and VCFs
    fasta = pyfaidx.Fasta(config.REF_GENOME)
    input_vcf = pysam.VariantFile(args.input_vcf)
    
    # Create a new VCF header with the new INFO field
    new_header = input_vcf.header
    new_header.info.add('DNABERT_SCORE', '1', 'Float', 'Confidence score from DNABERT-2 model')
    
    output_vcf = pysam.VariantFile(config.PREDICTION_OUTPUT_VCF, 'w', header=new_header)

    # 3. Iterate over VCF records and predict
    for record in tqdm(input_vcf.fetch(), desc="Predicting SVs"):
        try:
            sv_type = record.info.get("SVTYPE")
            if not sv_type or sv_type not in ['DEL', 'INS', 'DUP', 'INV']:
                output_vcf.write(record) # Write unchanged
                continue

            chrom, start, end = record.chrom, record.start, record.stop
            sv_len_info = record.info.get("SVLEN")
            sv_len = abs(sv_len_info[0] if isinstance(sv_len_info, tuple) else sv_len_info)
            
            # Construct sequences
            ref_seq = fasta[chrom][start - config.FLANKING_SIZE : end + config.FLANKING_SIZE].seq
            alt_seq = get_alt_sequence(fasta, chrom, start, record.ref, record.alts[0], sv_type, sv_len, end)

            if not ref_seq or not alt_seq:
                output_vcf.write(record)
                continue

            # Predict confidence
            score = predict(model, tokenizer, device, ref_seq, alt_seq)
            
            # Add score to record and write to new VCF
            record.info['DNABERT_SCORE'] = score
            output_vcf.write(record)

        except Exception as e:
            logger.warning(f"Could not process record {record.id}: {e}")
            output_vcf.write(record) # Write unchanged on error

    input_vcf.close()
    output_vcf.close()
    logger.info(f"Prediction complete. Annotated VCF saved to {config.PREDICTION_OUTPUT_VCF}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict SV confidence using a trained DNABERT-2 model.")
    parser.add_argument("--input_vcf", type=str, required=True, help="Path to the input VCF file for prediction.")
    parser.add_argument("--model_path", type=str, default=config.MODEL_SAVE_PATH, help="Path to the trained model .pth file.")
    args = parser.parse_args()
    main(args)