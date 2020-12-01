
tune_hyperparams = function(dir_name, X, Y, mask_x, mask_y, g, covars_r_x, covars_r_y, learn_r, Ignorable,
                            family, link, normalize){
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
  sigma="elu"; hs=c(64L,128L); bss=c(200L); lrs=c(0.001,0.01); impute_bs = bss[1]; arch="IWAE"
  niws=20L; n_epochss=2002L; n_hidden_layers = c(1L, 2L)
  
  # misc fixed params:
  add_miss_term = F; draw_miss = T; pre_impute_value = 0; sigma="elu"
  n_hidden_layers_r=0; h3=0  # no hidden layers in decoder_r, no nodes in that hidden layer (doesn't matter what h3 is)
  phi0=NULL; phi=NULL # only input when using logistic regression (known coefs)
  
  if(normalize){
    norm_means_x=colMeans(Xs$train, na.rm=T); norm_sds_x=apply(Xs$train,2,function(y) sd(y,na.rm=T))
    norm_mean_y=colMeans(Ys$train, na.rm=T); norm_sd_y=apply(Ys$train,2,function(y) sd(y,na.rm=T))
    dir_name = sprintf("%s/with_normalization",dir_name)
  }else{
    norm_means_x = rep(0, P); norm_sds_x = rep(1, P)
    norm_mean_y = 0; norm_sd_y = 1
  }
  
  LBs_trainVal = matrix(nrow = length(hs)*length(bss)*length(lrs)*length(niws)*length(n_epochss)*length(n_hidden_layers),
                        ncol = 12)
  colnames(LBs_trainVal) = c("h","bs","lr","niw","epochs","nhls","LB_train","MSE_train_x","MSE_train_y",
                             "LB_valid","MSE_valid_x","MSE_valid_y")
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
  
  res_test = dlglm(np$array(Xs$test), np$array(Rxs$test), np$array(Ys$test), np$array(Rys$test),
                    np$array(covars_r_x), np$array(covars_r_y),
                    np$array(norm_means_x), np$array(norm_sds_x), np$array(norm_mean_y), np$array(norm_sd_y),
                    learn_r, Ignorable, family, link,
                    impute_bs, arch, add_miss_term, draw_miss, 
                    train_params$pre_impute_value, train_params$n_hidden_layers, train_params$n_hidden_layers_r, 
                    train_params$h1, train_params$h2, train_params$h3, phi0, phi, 
                    0, saved_model, sigma, train_params$bs, 2L,
                    train_params$lr, train_params$L, train_params$M, trace=T)
  
  fixed.params = list(dir_name=dir_name, covars_r_x=covars_r_x, covars_r_y=covars_r_y, learn_r=learn_r, Ignorable=Ignorable, family=family, link=link)
  
  return(list(results=res_test, fixed.params=fixed.params))
  
}

