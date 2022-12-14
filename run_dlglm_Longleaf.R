# setwd("/pine/scr/d/e/deelim/out/Research/P3/")
source("dlglm.R")
source("run_mice.R")
source("run_dlglm_fx.R")
source("simulateData.R")


################ example #################
# Ns = c(1e5); Ps = c(50); Ds = c(2); phi0s = c(5)   # Ds: dimensionality of Z
# beta = 0.25; pi = 0.3; miss_pct_features = 50   # pi: % missingness in a miss_col. m_p_f: % of missing features
# NLs = c(F,F,F)  # linear generation of X from Z, Y from X, and R from X and Y
# mechanisms=c("MCAR","MAR","MNAR")
# sim_indexes = 1:5
# 
# ## data_types_x = c("real","cat","count","pos")
# ## vars = # in each data_types_x
# # data_types_x = c( rep("real",P) )
# vars = c(P,0,0,0)
# family="Multinomial"## family(link) = Gaussian(identity), Multinomial(mlogit), or Poisson(log)
# Cx = NULL; Cy = 2  # number of classes of X and Y
# 
# covars = "ally" ## covariates in missingness model ("obs","obsy","miss","missy","all","ally"). obs/miss/all X and/or y
# 
# normalize=F
# run_methods=c("dlglm","idlglm","mice")
##########################################

# sbatch -p volta-gpu --gres=gpu:1 --qos=gpu_access -N 1 -n 1 --mem=32g -t 8- -J "test_dlglm" --wrap="Rscript --no-save --no-restore --verbose run_dlglm_Longleaf.R 100000 50 2 5 NA 2 0.25 x 0.3 50 F,F,F MNAR,MAR,MCAR 2 50,0,0,0 Multinomial ally F dlglm,idlglm,mice > test_dlglm 2>&1" 


args = (commandArgs(TRUE))
print(args)
## args: 1) Ns, 2) Ps, 3) Ds, 4) phi_0s, 5) Cx, 6) Cy, 7) beta, 8) case 9) pi, 10) miss_pct_features,
##       11) NLs, 12) mechanisms, 13) sim_indexes, 14) vars, 15) family, 16) covars, 17) normalize, 18) run_methods, 19) init_r default or alt

args_types = c("numvec","numvec","numvec","numvec","num","num","num","chr","num","num",
               "boolvec","chrvec","numvec","numvec","chr","chr","bool","chrvec", "chr")
process_args = function(type, arg){
  if(grepl("vec",type)){
    ## converts to splitted strings
    arg0 = strsplit(arg,",")[[1]]
  }else{ arg0 = arg }
  
  if(grepl("num",type)){
    arg0 = as.numeric(arg0)
  }else if(grepl("bool",type)){
    arg0 = as.logical(arg0)
  }  ## for characters, nothing needs to be done
  
  return(arg0)
}

args2 = list()
for(j in 1:length(args)){
  args2[[j]] = process_args(type=args_types[j], arg = args[[j]])
  print(paste("args2",j,": ", args2[[j]]))
}
names(args2) = c("Ns", "Ps", "Ds", "phi0s", "Cx", "Cy", "beta", "case", "pi", "miss_pct_features", "NLs",
                 "mechanisms", "sim_indexes", "vars", "family", "covars", "normalize", "run_methods", "init_r")

family = args2$family
link = if(family=="Gaussian"){ "identity"
} else if(family=="Multinomial"){ "mlogit"
} else if(family=="Poisson"){ "log" }

args2$"link" = link


# run_dlglm_fx(Ns, Ps, Ds, phi0s, Cx, Cy, family, link, vars, beta, case, pi, miss_pct_features, normalize,
#              NLs, covars, mechanisms, sim_indexes, run_methods)

do.call(run_dlglm_fx,args2)     # named list of inputs














