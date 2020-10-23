
tune_hyperparams = function(dir_name, X, Y, mask_x, mask_y, g, covars_r_x, covars_r_y, learn_r, Ignorable,
                            family, link){
  # (family, link) = (Gaussian, identity), (Multinomial, mlogit), (Poisson, log)
  library(reticulate)
  
  np = import("numpy")
  torch = import("torch")
  source_python("dlglm.py")
  P = ncol(X); N = nrow(X)
  
  Xs = split(data.frame(X), g)        # split by $train, $test, and $valid
  Ys = split(data.frame(Y), g)        # split by $train, $test, and $valid
  Rxs = split(data.frame(mask_x), g)
  Rys = split(data.frame(mask_y), g)

  # dim_z --> as.integer() does floor()
  sigma="elu"; hs=c(64L,128L); bss=c(10000L); lrs=c(0.001,0.01); impute_bs = bss[1]; arch="IWAE"
  niws=5L; n_epochss=2002L; n_hidden_layers = c(1L, 2L)
  
  # misc fixed params:
  add_miss_term = F; draw_miss = T; pre_impute_value = 0; sigma="elu"
  n_hidden_layers_r=0; h3=0  # no hidden layers in decoder_r, no nodes in that hidden layer (doesn't matter what h3 is)
  phi0=NULL; phi=NULL # only input when using logistic regression (known coefs)
  
  norm_means_x=colMeans(Xs$train, na.rm=T); norm_sds_x=apply(Xs$train,2,function(y) sd(y,na.rm=T))
  norm_mean_y=colMeans(Ys$train, na.rm=T); norm_sd_y=apply(Ys$train,2,function(y) sd(y,na.rm=T))
  
  LBs_trainVal = matrix(nrow = length(hs)*length(bss)*length(lrs)*length(niws)*length(n_epochss)*length(n_hidden_layers),
                        ncol = 12)
  index = 1
  
  for(i in 1:length(hs)){for(j in 1:length(bss)){for(k in 1:length(lrs)){
    for(m in 1:length(niws)){for(mm in 1:length(n_epochss)){for(nn in 1:length(n_hidden_layers)){
      # dl.glm(np$array(Xs$valid), np$array(Rxs$valid), np$array(Ys$valid), np$array(Rys$valid),
      #                    np$array(covars_r_x), np$array(covars_r_y),
      #                    np$array(norm_means_x), np$array(norm_sds_x), np$array(norm_mean_y), np$array(norm_sd_y),
      #                    learn_r, Ignorable=F, family="Gaussian", link="identity",
      #                    impute_bs=NULL, arch="IWAE", add_miss_term=F, draw_miss=T, 
      #                    pre_impute_value=0, n_hidden_layers=2, n_hidden_layers_r=0, 
      #                    h1=8, h2=8, h3=0, phi0=NULL, phi=NULL, dec_distrib="Normal",
      #                    train=1, saved_model=NULL, sigma="elu", bs = 64, n_epochs = 2002,
      #                    lr=0.001, L=20, M=20, trace=F)
      
      res_train = dlglm(np$array(Xs$train), np$array(Rxs$train), np$array(Ys$train), np$array(Rys$train),
                         np$array(covars_r_x), np$array(covars_r_y),
                         np$array(norm_means_x), np$array(norm_sds_x), np$array(norm_mean_y), np$array(norm_sd_y),
                         learn_r, Ignorable, family, link,
                         impute_bs, arch, add_miss_term, draw_miss, 
                         pre_impute_value, n_hidden_layers[nn], n_hidden_layers_r, 
                         hs[i], hs[i], h3, phi0, phi, 
                         1, NULL, sigma, bss[j], n_epochss[mm],
                         lrs[k], niws[m], niws[m], trace=T)
      res_valid = dlglm(np$array(Xs$valid), np$array(Rxs$valid), np$array(Ys$valid), np$array(Rys$valid),
                         np$array(covars_r_x), np$array(covars_r_y),
                         np$array(norm_means_x), np$array(norm_sds_x), np$array(norm_mean_y), np$array(norm_sd_y),
                         learn_r, Ignorable, family, link,
                         impute_bs, arch, add_miss_term, draw_miss, 
                         pre_impute_value, n_hidden_layers[nn], n_hidden_layers_r, 
                         hs[i], hs[i], h3, phi0, phi, 
                         0, res_train$saved_model, sigma, bss[j], 2L,
                         lrs[k], niws[m], niws[m], trace=T)
      
      
      LBs_trainVal[index,]=c(hs[i],bss[j],lrs[k],niws[m],n_epochss[mm],
                             n_hidden_layers[nn],
                             res_train$'LB', res_train$'MSE'$miss_x[length(res_train$'MSE'$miss_x)],
                             res_train$'MSE'$miss_y[length(res_train$'MSE'$miss_y)],
                             res_valid$'LB', res_valid$'MSE'$miss_x[length(res_valid$'MSE'$miss_x)],
                             res_valid$'MSE'$miss_y[length(res_valid$'MSE'$miss_y)])
      
      print(LBs_trainVal)
      if(is.na(res_valid$'LB')){res_valid$'LB'=-Inf}
      
      # save only the best result currently (not all results) --> save memory
      if(index==1){opt_LB = res_valid$'LB'; save(res_train, file=sprintf("%s/temp_opt_train.out",dir_name)); torch$save(res_train$'saved_model',sprintf("%s/temp_opt_train_saved_model.pth",dir_name))  #; save(opt_train, file="temp_opt_train.out")
      }else if(res_valid$'LB' > opt_LB){opt_LB = res_valid$'LB'; save(res_train, file=sprintf("%s/temp_opt_train.out",dir_name)); torch$save(res_train$'saved_model',sprintf("%s/temp_opt_train_saved_model.pth",dir_name))} #; save(opt_train, file="temp_opt_train.out")
      
      rm(res_train)
      rm(res_valid)
      index=index+1
      
      # release gpu memory
      reticulate::py_run_string("import torch")
      reticulate::py_run_string("torch.cuda.empty_cache()")
    }}}}}}
  
  saved_model = torch$load(sprintf("%s/temp_opt_train_saved_model.pth",dir_name))
  load(sprintf("%s/temp_opt_train.out",dir_name))
  train_params=res_train$train_params
  
  res_test = dlglm(np$array(Xs$valid), np$array(Rxs$valid), np$array(Ys$valid), np$array(Rys$valid),
                    np$array(covars_r_x), np$array(covars_r_y),
                    np$array(norm_means_x), np$array(norm_sds_x), np$array(norm_mean_y), np$array(norm_sd_y),
                    learn_r, Ignorable, family, link,
                    impute_bs, arch, add_miss_term, draw_miss, 
                    train_params$pre_impute_value, train_params$n_hidden_layers, train_params$n_hidden_layers_r, 
                    train_params$h1, train_params$h2, train_params$h3, phi0, phi, 
                    0, saved_model, sigma, train_params$bs, 2L,
                    train_params$lr, train_params$L, train_params$M, trace=T)
  
  return(res_test)
  
}

N=1e5; P=8; data_types=rep("real",P); family="Gaussian"; link="identity"
dataset = sprintf("SIM_N%d_P%d_X%s_Y%s", N, P, data_types[1], family)
case = "x"; pi=0.5

learn_r = T; covars_r_x = rep(1,P); covars_r_y = 1  # include all
Ignorable=F

mechanisms=c("MCAR","MAR","MNAR"); sim_indexes = 1:5
# mechanism="MNAR"; sim_index=1
for(m in 1:length(mechanisms)){for(s in 1:length(sim_indexes)){
  
  dir_name = sprintf("Results/%s/miss_%s/sim%d", dataset, case, sim_indexes[s])   # to save interim results
  fname = sprintf("%s/res_dlglm_%s_%d.RData",dir_name,mechanisms[m],pi*100)
  if(!file.exists(fname)){
    load( sprintf("%s/data_%s_%d.RData", dir_name, mechanisms[m], pi*100) )  # loads "X","Y","mask_x","mask_y","g"
    
    dir_name0 = sprintf("%s/%s_%d", dir_name, mechanisms[m], pi*100)
    ifelse(!dir.exists(dir_name0), dir.create(dir_name0), F)
    res = tune_hyperparams(dir_name0, X, Y, mask_x, mask_y, g,
                           covars_r_x, covars_r_y, learn_r, Ignorable,
                           family, link)
    save(res, file=fname)
  } else{
    next
  }
}}
