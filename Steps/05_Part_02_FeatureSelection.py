#!/usr/bin/env python3
import scanpy as sc
import numpy as np
import seaborn as sns
import logging
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def main(args):
	outdir = Path(args.output_dir)
	outdir.mkdir(parents=True, exist_ok=True)
	log = outdir / f"{args.sample}_step05_part2_FeatureSelection.log"
	logging.basicConfig(level=logging.INFO, handlers=[
		logging.FileHandler(str(log)),
		logging.StreamHandler()
	], format="%(asctime)s %(levelname)s: %(message)s")
	
	logging.info("Continuing step 5: Feature selection")
	

	# Import data
	input_path = Path(args.input)
	if not input_path.exists():
		logging.error(f"Input path not found: {input_path}")
		raise SystemExit(1)
		
	adata = sc.read_h5ad(str(input_path))
	logging.info(f"Read in data from step 5 part 1")
	
	# Retrieve deviance that was computed in part 1
	dev_scores = adata.var["binomial_deviance"].values

	# Select top 2,000 highly deviant genes
	logging.info(f"Selecting top 2,000 highly deviant genes")
	idx = dev_scores.argsort()[-2000:]
	mask = np.zeros(adata.var_names.shape, dtype=bool)
	mask[idx] = True

	adata.var["highly_deviant"] = mask
	#adata.var["binomial_deviance"] = binomial_deviance

	sc.pp.highly_variable_genes(adata, layer="log1p_norm")

	# Generate Plot
	logging.info(f"Generating scatterplot: dispersions vs means")
	feat = outdir/ f"{args.sample}_feature_selection_scatterplot.png"
	
	fig, ax = plt.subplots(figsize=(6,5))
	sns.scatterplot(
    data=adata.var, x="means", y="dispersions", hue="highly_deviant", s=5
	)
	plt.savefig(feat)
	plt.close()
	logging.info(f"Saved selected feature scatterplot: {feat}")

	# Save data
	out_h5 = outdir / f"{args.sample}_normalized_data.h5ad"
	adata.write(out_h5)
	logging.info(f"Wrote normalized AnnData: {out_h5}")
	
if __name__=="__main__":
	parser = argparse.ArgumentParser(description = "Feature Selection of Data")
	parser.add_argument("--input", required = True, help = "Path to step 5 part 1 h5ad output file")
	parser.add_argument("--output-dir", required = True, help = "Directory to write output files")
	parser.add_argument("--sample", required = True, help = "Data set name")
	args = parser.parse_args()
	main(args)
