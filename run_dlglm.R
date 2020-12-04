source("dlglm.R")
# data_types: real, cat, cts; family="Gaussian (identity)", "Multinomial (mlogit)", "Poisson (log)"
N=1e5; P=8; data_types=rep("real",P)
family="Gaussian"; link="identity"
# family="Multinomial"; link="mlogit"
mu=0; sd=1; beta=5
case = "x"; pi=0.5; phi0=100
learn_r = T; covars_r_x = rep(1,P); covars_r_y = 1  # include all
Ignorable=F; normalize=F
mechanisms="MNAR"; sim_indexes = 1

data_type_x = data_types[1]
data_type_y = if(family=="Gaussian"){"real"}else if(family=="Multinomial"){"cat"}else if(family=="Poisson"){"cts"}

prefix = sprintf("Xmean%dsd%d_beta%d_pi%d/",
                 mu, sd, beta, pi*100)
dataset = sprintf("SIM_N%d_P%d_X%s_Y%s", N, P, data_types[1], family)

for(m in 1:length(mechanisms)){for(s in 1:length(sim_indexes)){
  
  dir_name = sprintf("Results_X%s_Y%s/%s%s/miss_%s/phi%d/sim%d", data_type_x,data_type_y,prefix, dataset, case, phi0, sim_indexes[s])   # to save interim results
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




# sbatch -p gpu --gres=gpu:1 --qos=gpu_access -N 1 -n 1 --mem=24g -t 3- -o /pine/scr/d/e/deelim/dump/dlglm.out -J sim_dlglm --wrap='R CMD BATCH run_dlglm.R'