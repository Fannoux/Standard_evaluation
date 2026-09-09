python heritability_cka.py \
  --relmat-bin genetic_data/input.grm.bin --relmat-id genetic_data/input.grm.id \
  --feats CNN=genetic_data/CNN_gcta.pheno_formatted_norm \
          RegionProps=genetic_data/RegionProps_gcta.pheno_formatted_norm \
          ShapeEmbed=genetic_data/shapeembed_gcta.pheno_formatted_norm \
          VAE=genetic_data/vae_gcta.pheno_formatted_norm \
          adjSC=genetic_data/adjSC_gcta.pheno_formatted_norm \
  --out results_heritability_nocov
  #--covariates genetic_data/gcta.cov \
