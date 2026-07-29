#!/usr/bin/env python3
import argparse
from pathlib import Path
import logging
import scanpy as sc
import doubletdetection
from scipy.sparse import csr_matrix

def main(args):
	outdir = Path(args.output_dir)
	outdir.mkdir(parents=True, exist_ok=True)
	log = outdir / f"{args.sample}_step03.log"
	logging.basicConfig(level=logging.INFO, handlers=[
		logging.FileHandler(str(log)),
		logging.StreamHandler()
	], format="%(asctime)s %(levelname)s: %(message)s")
	logging.info("Starting step 3: doublet detection")
	
	# Read Data
	logging.info("Running scrublet...")
	adata=sc.read_h5ad(args.input)

	# Scrublet
	sc.pp.scrublet(adata)
	
	adata.obs['scrublet_doublet'] = adata.obs['predicted_doublet']
	adata.obs['scrublet_score'] = adata.obs['doublet_score']
	
	del adata.obs['predicted_doublet']
	del adata.obs['doublet_score']
	
	# Doublet Detection
	logging.info("Running doublet detection...")
	clf = doubletdetection.BoostClassifier(
		n_iters=10,
		clustering_algorithm="leiden",
		standard_scaling=True,
		pseudocount=0.1,
		n_jobs=int(args.threads)
	)
		
	doublets = clf.fit(adata.X).predict(p_thresh=1e-16, voter_thresh=0.5)
	doublet_score=clf.doublet_score()
	
	adata.obs['doubletdetection_doublet'] = doublets.astype(bool)
	adata.obs['doubletdetection_score'] = doublet_score
	
	out_h5 = outdir / f"{args.sample}_Post_QualityControl.h5ad"
	adata.write(out_h5)
	logging.info(f"Wrote final post quality control: {out_h5}")
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Detect doublets with scrublet")
	parser.add_argument("--input", required = True, help = "Path to step 2 filtered h5ad")
	parser.add_argument("--output-dir", required = True, help = "Directory to write outputs")
	parser.add_argument("--sample", required = True, help = "Sample name")
	parser.add_argument("--threads", required = True, help = "Threads")
	args = parser.parse_args()
	main(args)