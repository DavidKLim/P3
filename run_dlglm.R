source("dlglm.R")
N=1e5; P=8; data_types=rep("real",P); family="Gaussian"; link="identity"
mu=0; sd=1; beta=5
case = "x"; pi=0.5; phi0=100
learn_r = T; covars_r_x = rep(1,P); covars_r_y = 1  # include all
Ignorable=F; normalize=F
mechanisms="MNAR"; sim_indexes = 1




prefix = sprintf("Xmean%dsd%d_beta%d_pi%d/",
                 mu, sd, beta, pi*100)
dataset = sprintf("SIM_N%d_P%d_X%s_Y%s", N, P, data_types[1], family)
# mechanism="MNAR"; sim_index=1
for(m in 1:length(mechanisms)){for(s in 1:length(sim_indexes)){
  
  dir_name = sprintf("Results/%s%s/miss_%s/phi%d/sim%d", prefix, dataset, case, phi0, sim_indexes[s])   # to save interim results
  fname = sprintf("%s/res_dlglm_%s_%d.RData",dir_name,mechanisms[m],pi*100)
  if(!file.exists(fname)){
    load( sprintf("%s/data_%s_%d.RData", dir_name, mechanisms[m], pi*100) )  # loads "X","Y","mask_x","mask_y","g"
    
    dir_name0 = sprintf("%s/%s_%d", dir_name, mechanisms[m], pi*100)
    ifelse(!dir.exists(dir_name0), dir.create(dir_name0), F)
    res = tune_hyperparams(dir_name0, X, Y, mask_x, mask_y, g,
                           covars_r_x, covars_r_y, learn_r, Ignorable,
                           family, link, normalize)
    save(res, file=fname)
  } else{
    next
  }
}}

