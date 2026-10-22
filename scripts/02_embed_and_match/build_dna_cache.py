"""Build the DNA embedding cache that training and evaluation read.

For every perturbed gene: take the 524,288 bp GRCh38 window centred on its canonical TSS, run the
frozen Flashzoi/Borzoi trunk to 128 bp-resolution features (4096 x 1536), build the +/-200 bp TSS
perturbation mask at the same resolution, and store both - with the cCRE intervals inside the
window when a cCRE BED file is given - in a DiskDNACache keyed by the upper-case gene symbol.

Gene symbols come from --genes (a text file, one per line) or from the `perturbation` column of the
OT index files in --idx_dir (labels `<GENE>-<dname>`).

    python build_dna_cache.py --idx_dir "$DATA_ROOT/ot_ids/line_specific/k562" --dna_cache_dir "$DNA_CACHE_DIR"
"""
import argparse
import glob
import os
import sys
import traceback

import pandas as pd
import torch
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.diffusion_ot.data import preprocessing
from src.diffusion_ot.data.cache import DiskDNACache
from src.diffusion_ot.utils.dna_utils import (
    aggregate_mask_perturbation,
    gen_gene_TSS_sequence,
    get_dna_record,
    load_ccres_bed,
)

try:
    from baskerville import dna  # calico/baskerville
except ImportError:  # vendored copy of dna_1hot
    from src.diffusion_ot.data import _dna_onehot as dna


def borzoi_forward_full(borzoi_model, dna_1hot_batch, device):
    """Borzoi trunk to full-resolution features at 128 bp tokens.

    dna_1hot_batch: (B, 4, 524288) float tensor on device. Returns (B, 4096, 1536) float32.
    """
    with torch.no_grad():
        feat = borzoi_model.conv_dna(dna_1hot_batch)
        feat = borzoi_model._max_pool(feat)
        for block in borzoi_model.res_tower:
            feat = block(feat)
        feat = borzoi_model.unet1(feat).transpose(1, 2)
        # transformer in bf16 for speed
        if device.type == 'cuda':
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                feat = feat.to(torch.bfloat16)
                feat = borzoi_model.transformer(feat)
            feat = feat.to(torch.float32)
        else:
            feat = borzoi_model.transformer(feat)
            feat = feat.to(torch.float32)
    return feat


def load_annotation_by_ensembl(annotation_path):
    """Transcript rows of a GENCODE GTF, grouped by Ensembl gene ID without version.

    gen_gene_TSS_sequence reads the first nine GTF columns by position and is called with the
    HGNC Ensembl ID, which carries no version suffix.
    """
    annot = pd.read_csv(annotation_path, sep='\t', comment='#', header=None, usecols=range(9))
    annot = annot.rename(columns={0: 'chrom', 2: 'feature', 3: 'start', 4: 'end', 6: 'strand', 8: 'attr'})
    annot = annot[annot['feature'] == 'transcript']
    annot['gene_id'] = annot['attr'].str.extract(r'gene_id "([^".]+)', expand=False)
    return dict(tuple(annot.groupby('gene_id')))


def read_gene_symbols(genes_path=None, idx_dir=None):
    if genes_path:
        with open(genes_path) as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        labels = set()
        for path in sorted(glob.glob(os.path.join(idx_dir, '*_ot_match.*.idx.csv'))):
            labels.update(pd.read_csv(path, usecols=['perturbation'])['perturbation'])
        if not labels:
            raise ValueError(f"No *_ot_match.*.idx.csv files in {idx_dir}")
        symbols = [label.rsplit('-', 1)[0] for label in labels if not label.startswith('control-')]
    return sorted({s.upper() for s in symbols})


def build_dna_cache(gene_symbols, dna_cache_dir, borzoi_model, dna_record, gene_annot,
                    ccre_intervals_dict=None, promoter_len=200, perturb_type='CRISPR-i', device=None):
    """Write one cache entry per gene symbol; genes already in the cache are skipped."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    borzoi_model = borzoi_model.to(device).eval()

    ensembl_ids = preprocessing.convert_to_ensembl(gene_symbols, suffix_token=0, verbose=True)
    valid_pairs, failed = [], []
    for symbol, ensembl_id in zip(gene_symbols, ensembl_ids):
        if ensembl_id.startswith('ENSG'):
            valid_pairs.append((symbol, ensembl_id))
        else:
            failed.append((symbol, ensembl_id, "no Ensembl ID"))

    cache = DiskDNACache(dna_cache_dir, mode='w+', device=device)
    processed = 0
    print(f"Processing {len(valid_pairs)} genes ({len(failed)} without an Ensembl ID)", flush=True)
    for i, (symbol, ensembl_id) in enumerate(valid_pairs):
        if symbol in cache:
            continue
        try:
            seq_info = gen_gene_TSS_sequence(ensembl_id, dna_record=dna_record,
                                             all_annots_transcript=gene_annot, jitter_window=0,
                                             ccre_intervals_dict=ccre_intervals_dict)
            if seq_info is None or len(seq_info) == 0:
                failed.append((symbol, ensembl_id, "not in annotation"))
                continue
            seq_info[0]['promoter_len'] = promoter_len
            seq_info[0]['perturb_type'] = perturb_type

            seq_ref, mask_full = aggregate_mask_perturbation(seq_info[0])  # mask_full: (1, L)
            one_hot = dna.dna_1hot(seq_ref).T  # (4, L)
            dna_single = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0).to(device)
            feat = borzoi_forward_full(borzoi_model, dna_single, device)  # (1, 4096, 1536)

            mask_full_tensor = torch.tensor(mask_full, dtype=torch.float32).unsqueeze(0).to(device)
            mask_128 = F.interpolate(mask_full_tensor, size=feat.shape[1], mode='nearest')  # (1, 1, 4096)

            cache[symbol] = (feat[0].cpu(), mask_128[0].cpu(), seq_info[0].get('ccre_intervals_rel', []))
            processed += 1
        except Exception as e:
            failed.append((symbol, ensembl_id, str(e)))
            traceback.print_exc()
        if i % 50 == 0:
            cache.save_metadata()
            print(f"  {i}/{len(valid_pairs)}", flush=True)
    cache.save_metadata()

    print(f"Done: {processed} new entries, {len(cache.metadata)} in cache, {len(failed)} failed.")
    for symbol, ensembl_id, reason in failed[:20]:
        print(f"  failed {symbol} ({ensembl_id}): {reason}")
    return failed


def main():
    parser = argparse.ArgumentParser(description="Precompute the Borzoi DNA embedding cache.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--genes', help='Text file of gene symbols, one per line.')
    source.add_argument('--idx_dir', help='Directory of OT index files; genes are read from their perturbation column.')
    parser.add_argument('--dna_cache_dir', default=os.environ.get('DNA_CACHE_DIR'),
                        help='Output cache directory (default: $DNA_CACHE_DIR).')
    parser.add_argument('--reference', default=os.environ.get('DNA_REFERENCE_PATH'),
                        help='Directory of per-chromosome GRCh38 FASTA files (default: $DNA_REFERENCE_PATH).')
    parser.add_argument('--annotation', default=os.environ.get('GENE_ANNOTATION_PATH'),
                        help='GENCODE transcript GTF (default: $GENE_ANNOTATION_PATH).')
    parser.add_argument('--hgnc', default=os.environ.get('HGNC_COMPLETE_SET_PATH'),
                        help='HGNC hgnc_complete_set.txt for symbol -> Ensembl (default: $HGNC_COMPLETE_SET_PATH).')
    parser.add_argument('--ccre_bed', default=os.environ.get('CCRE_BED_PATH') or None,
                        help='ENCODE cCRE BED(.gz); stores cCRE intervals for the cCRE bias (default: $CCRE_BED_PATH).')
    parser.add_argument('--borzoi', default='johahi/flashzoi-replicate-0',
                        help='Hugging Face id of the Borzoi weights.')
    args = parser.parse_args()
    for name in ('dna_cache_dir', 'reference', 'annotation', 'hgnc'):
        if not getattr(args, name):
            parser.error(f"--{name} is required (or set its environment variable)")

    from borzoi_pytorch import Borzoi

    preprocessing.HGNC_COMPLETE_SET_PATH = args.hgnc
    genes = read_gene_symbols(args.genes, args.idx_dir)
    print(f"{len(genes)} gene symbols")
    dna_record = get_dna_record(args.reference)
    gene_annot = load_annotation_by_ensembl(args.annotation)
    ccre_intervals_dict = load_ccres_bed(args.ccre_bed) if args.ccre_bed else None
    borzoi_model = Borzoi.from_pretrained(args.borzoi)
    build_dna_cache(genes, args.dna_cache_dir, borzoi_model, dna_record, gene_annot,
                    ccre_intervals_dict=ccre_intervals_dict)


if __name__ == "__main__":
    main()
