library(reticulate)

run_dlglm_fx = function(Ns, Ps, Ds, phi0s, Cx, Cy, family, link, vars, beta, case, pi, miss_pct_features, normalize,
                     NLs, covars, mechanisms, sim_indexes, run_methods = c("dlglm","mice"), init_r="default"){
  
  inputs = match.call()
  print(inputs)
  
  data_types_x = c(rep("real",vars[1]), rep("cat", vars[2]), rep("count", vars[3]), rep("pos", vars[4]))
  NL_x = NLs[1]; NL_y = NLs[2]; NL_r = NLs[3]
  
  
  data_type_x_pref = if(all(data_types_x == data_types_x[1])){ data_types_x[1] } else{ "mixed" }
  data_type_y_pref = if(family=="Gaussian"){"real"}else if(family=="Multinomial"){"cat"}else if(family=="Poisson"){"cts"}
  
  prefix = sprintf("Xr%dct%dcat%d_beta%f_pi%d/",
                   sum(data_types_x=="real"), sum(data_types_x=="count"), sum(data_types_x=="cat"),
                   beta, pi*100)
  
  trace=F; draw_miss=T; learn_r=T
  early_stop=T
  
  if(init_r=="default"){dlglm_pref = ""}else if(init_r=="alt"){dlglm_pref="alt_init/"}
  
  
                       
  for(a in 1:length(Ns)){for(b in 1:length(Ps)){for(c in 1:length(Ds)){for(d in 1:length(phi0s)){for(m in 1:length(mechanisms)){for(s in 1:length(sim_indexes)){
    
    
    dataset = sprintf("SIM_N%d_P%d_D%d", Ns[a], Ps[b],Ds[c])
    
    iNL_x = if(NL_x){"NL"}else{""}
    iNL_y = if(NL_y){"NL"}else{""}
    iNL_r = if(NL_r){"NL_"}else{""}
    dir_name = sprintf("Results_%sX%s_%sY%s/%s%s/miss_%s%s/phi%d/sim%d", iNL_x, data_type_x_pref, iNL_y, data_type_y_pref, prefix, dataset, iNL_r, case, phi0s[d], sim_indexes[s])
    
    fname = sprintf("%s/%sres_dlglm_%s_%d_%d.RData",dir_name,dlglm_pref,mechanisms[m],miss_pct_features,pi*100)
    ifname = sprintf("%s/Ignorable/res_dlglm_%s_%d_%d.RData",dir_name,mechanisms[m],miss_pct_features,pi*100)
    fname_mice = sprintf("%s/res_mice_%s_%d_%d.RData",dir_name,mechanisms[m],miss_pct_features,pi*100)
    fname_miwae = sprintf("%s/res_miwae_%s_%d_%d.RData",dir_name,mechanisms[m],miss_pct_features,pi*100)
    fname_notmiwae = sprintf("%s/res_notmiwae_%s_%d_%d.RData",dir_name,mechanisms[m],miss_pct_features,pi*100)
    
    data_fname = sprintf("%s/data_%s_%d_%d.RData", dir_name, mechanisms[m], miss_pct_features, pi*100)
    params_fname = sprintf("%s/params_%s_%d_%d.RData", dir_name, mechanisms[m], miss_pct_features, pi*100)
    sim.params = list(N=Ns[a], D=Ds[c], P=Ps[b], data_types=data_types_x, family=family, sim_index=sim_indexes[s], ratios=c(train=.8,valid=.1,test=.1),
                      beta=beta, C=Cx, Cy=Cy, NL_x=NL_x, NL_y=NL_y)
    miss.params = list(scheme="UV", mechanism=mechanisms[m], NL_r=NL_r, pi=pi, phi0=phi0s[d], miss_pct_features=miss_pct_features,
                       miss_cols=NULL, ref_cols=NULL)
    
    print(data_fname)
    if(!file.exists(data_fname)){
      print("No existing simulated data found. Simulating data...")
      dat = prepareData(sim.params = sim.params,
                  miss.params=miss.params,
                  case=case)
      X = dat$X; Y = dat$Y; mask_x = dat$mask_x; mask_y = dat$mask_y; g = dat$g
      # sim.params = dat$sim.params; miss.params = dat$miss.params
      # return(list(X=X,Y=Y,mask_x=mask_x,mask_y=mask_y,g=g,sim.params=sim.params,miss.params=miss.params,sim.data=sim.data,sim.mask=sim.mask))
    }else{
      print("Loading existing simulated data")
      load( data_fname )  # loads "X","Y","mask_x","mask_y","g"
      load( params_fname )  # sim.params, miss.params, sim.data, sim.mask
    }
    
    #### hyperparams simulations
    hyperparams = list(sigma="elu", bss=c(1000L), lrs=c(0.01,0.001), impute_bs = 1000L, arch="IWAE",
                       niws=5L, n_imps = 500L, n_epochss=2002L, n_hidden_layers = c(0L,1L,2L), n_hidden_layers_y = c(0L), n_hidden_layers_r = c(0L,1L),
                       h=c(128L,64L), h_y=NULL, h_r=c(16L,32L),
                       dim_zs = c(as.integer(floor(ncol(X)/12)),as.integer(floor(ncol(X)/4)), as.integer(floor(ncol(X)/2)), as.integer(floor(3*ncol(X)/4))),
                       L1_weights = 0)
    
    dir_name0 = sprintf("%s/%s%s_%d_%d", dir_name, dlglm_pref, mechanisms[m], miss_pct_features, pi*100)
    idir_name0 = sprintf("%s/Ignorable/%s_%d_%d", dir_name,mechanisms[m], miss_pct_features, pi*100)
    
    
    covars_r_y = if(grepl("y",covars)){ 1L }else{ 0L }
    covars_r_x = if(grepl("miss",covars)){ (colMeans(mask_x)!=1)^2    # SELECT COVARIATES FOR MNAR
    } else if(grepl("obs", covars)){ (colMeans(mask_x)==1)^2 
    } else if(grepl("all", covars)){ covars_r_x = rep(1L,Ps[b]) }
    
    if("dlglm" %in% run_methods){ ifelse(!dir.exists(dir_name0), dir.create(dir_name0, recursive=T), F) }
    if("idlglm" %in% run_methods){ ifelse(!dir.exists(idir_name0), dir.create(idir_name0, recursive=T), F) }
    
    if(!file.exists(fname) & ("dlglm" %in% run_methods)){
      res = dlglm(dir_name0, X, Y, mask_x, mask_y, g,
                  covars_r_x, covars_r_y, learn_r, data_types_x, Ignorable=F,
                  family, link, normalize, early_stop, trace, draw_miss, init_r=init_r, hyperparams=hyperparams)
      save(res, file=fname)
      rm(res)
    } 
    
    if(!file.exists(ifname) & ("idlglm" %in% run_methods)){
      res = dlglm(idir_name0, X, Y, mask_x, mask_y, g,
                  covars_r_x, covars_r_y, learn_r, data_types_x, Ignorable=T,
                  family, link, normalize, early_stop, trace, draw_miss, init_r="default", hyperparams=hyperparams)
      save(res, file=ifname)
      rm(res)
    } 
    
    if(!file.exists(fname_mice) & ("mice" %in% run_methods)){
      source("run_mice.R")
      res_mice = run_mice(dir_name, sim.params, miss.params, X, Y, mask_x, mask_y, g, data_types_x, family, niws = 500)
      save(res_mice, file=fname_mice)
      rm(res_mice)
    }
    
    
    
    if(!file.exists(fname_miwae) & ("miwae" %in% run_methods)){
      source("run_miwae.R")
      res_miwae = run_miwae(dir_name, X, Y, mask_x, mask_y, g, hyperparams, niws = 500L)
      save(res_miwae, file=fname_miwae)
      rm(res_miwae)
    }
    
    if(!file.exists(fname_notmiwae) & ("notmiwae" %in% run_methods)){
      source("run_miwae.R")
      res_notmiwae = run_notmiwae(dir_name, X, Y, mask_x, mask_y, g, mprocess="linear", hyperparams, niws = 500L)
      save(res_notmiwae, file=fname_notmiwae)
      rm(res_notmiwae)
    }
  }}}}}}
}
