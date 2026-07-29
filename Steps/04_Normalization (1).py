#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def setup_logger(outdir: Path, prefix: str) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / f"{prefix}_step04_ConcatenateNormalize.log"

    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    fh = logging.FileHandler(str(log_path))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main(args):
    samples = []
    with open(args.samples, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Each non-comment line must have at least 2 fields; got: {line}")
            samples.append(parts[1])

    outdir = Path(args.output_dir)
    logger = setup_logger(outdir, args.output_prefix)

    logger.info(f"Loading up to {len(samples)} h5ad files and concatenating...")
    adatas = []
    loaded_samples = []
    for s in samples:
        in_path = Path(args.file_path.format(sample=s))
        if not in_path.exists():
            logger.warning(f"Missing h5ad for sample {s}; skipping {in_path}")
            continue

        logger.info(f"reading {s}: {in_path}")
        adatas.append(sc.read_h5ad(str(in_path)))
        loaded_samples.append(s)

    if len(adatas) < 2:
        raise RuntimeError(f"After skipping missing files, only found {len(adatas)} sample(s); need at least 2 to concatenate.")

    adata = sc.concat(
        adatas,
        label="sample",
        keys=loaded_samples,
        index_unique="-",
    )
    logger.info("Concatenation complete.")
    logger.info(f"Result: {adata.n_obs} cells, {adata.n_vars} genes")

    raw_out = outdir / f"{args.output_prefix}_raw_concatenated.h5ad"
    logger.info(f"Writing raw concatenated output: {raw_out}")
    adata.write(raw_out)
    
    # Pre-normalization histogram (Using the clean upstream total_counts)
    logger.info("Writing pre-normalization histogram...")
    plot_path = outdir / f"{args.output_prefix}_pre_normalization_histogram.png"
    
    # Use standard raw total counts for the x-axis
    p1 = sns.displot(adata.obs["total_counts"], bins=100, kde=False)
    p1.set_xlabels("Log1p UMI Counts per Cell")
    p1.set_ylabels("Number of Cells")
    p1.set(title="Pre-normalized Library Size per Cell")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Perform Shifted-Logarithm Normalization
    logger.info("Running scanpy: normalize_total (per cell) ...")
    scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)

    logger.info("Running scanpy: log1p ...")
    # log1p transform
    adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

    # Post-normalization histogram (Shifted Logarithm Profile)
    logger.info("Writing post-normalization histogram...")
    #post_profile = adata.X.sum(1)
    #post_profile = post_profile.A1 if hasattr(post_profile, "A1") else post_profile

    plot_path = outdir / f"{args.output_prefix}_post_normalization_histogram.png"
    p2 = sns.displot(adata.layers["log1p_norm"].sum(1), bins=100, kde=False)
    p2.set_xlabels("Shifted Logarithm Expression Sum (per cell)")
    p2.set_ylabels("Number of Cells")
    p2.set(title="Per Cell Shifted Logarithm Distribution")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Wrote post-normalization histogram: {plot_path}")

    out_h5 = outdir / f"{args.output_prefix}_normalized_concatenated.h5ad"
    logger.info(f"Writing output: {out_h5}")
    adata.write(out_h5)
    logger.info("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Concatenate multiple h5ad samples then normalize once (shared).")
    p.add_argument("--samples", required=True, help="Path to text file with sample names (2nd column).")
    p.add_argument("--file-path", required=True, help="Template path for each sample h5ad, must include '{sample}' placeholder.")
    p.add_argument("--output-prefix", required=True, help="Prefix for output files/logs.")
    p.add_argument("--output-dir", required=True, help="Directory to write output files.")
    p.add_argument("--target-sum", type=float, default=None, help="target_sum for normalize_total (None uses scanpy default).")
    args = p.parse_args()
    main(args)