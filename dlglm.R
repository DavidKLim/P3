
# packages: reticulate
dlglm = function(dir_name, X, Y, mask_x, mask_y, g, covars_r_x, covars_r_y, learn_r, data_types_x, Ignorable,
                            family, link, normalize, early_stop, trace, draw_miss=T,
                 hyperparams = list(sigma="elu", hs=c(128L,64L), bss=c(1000L), lrs=c(0.01,0.001), impute_bs = 10000L, arch="IWAE",
                                        niws=5L, n_imps = 500L, n_epochss=2002L, n_hidden_layers = c(0L,1L,2L), n_hidden_layers_y = c(0L), n_hidden_layers_r = c(0L,1L),
                                        dim_zs = c(as.integer(floor(ncol(X)/4)), as.integer(floor(ncol(X)/2)), as.integer(floor(3*ncol(X)/4))),
                                        L1_weights = 0)){
  
  ## Hyperparameters ##
  # # dim_z --> as.integer() does floor()
  # # sigma="elu"; hs=c(64L,128L); bss=c(200L); lrs=c(0.001,0.01); impute_bs = bss[1]; arch="IWAE"
  # # sigma="elu"; hs=c(64L,128L); bss=c(1000L); lrs=c(0.001,0.01); impute_bs = bss[1]; arch="IWAE"   # TEST. COMMENT OUT AND REPLACE W ABOVE LATER
  # # niws=5L; n_epochss=2002L; n_hidden_layers = c(1L, 2L); n_hidden_layers_y = 0L
  # sigma="elu"; hs=c(128L,64L); bss=c(10000L); lrs=c(0.01,0.001); impute_bs = bss[1]; arch="IWAE"   # TEST. COMMENT OUT AND REPLACE W ABOVE LATER
  # niws=5L; n_epochss=2002L; n_hidden_layers = c(1L,2L); n_hidden_layers_y = c(0L,1L); n_hidden_layers_r = c(0L,1L)
  # dim_zs = c(as.integer(floor(ncol(X)/4)), as.integer(floor(ncol(X)/2)), as.integer(floor(3*ncol(X)/4)))
  # # dim_zs = c(as.integer(floor(ncol(X)/4)), as.integer(floor(ncol(X)/2)))
  # # if(Ignorable){ L1_weights = 0 } else{ L1_weights = c(1e-1, 5e-2, 0) }
  # L1_weights=0
  sigma = hyperparams$sigma; hs = hyperparams$hs; bss = hyperparams$bss; lrs = hyperparams$lrs; impute_bs = hyperparams$impute_bs; arch = hyperparams$arch
  niws = hyperparams$niws; n_imps = hyperparams$n_imps; n_epochss = hyperparams$n_epochss; n_hidden_layers = hyperparams$n_hidden_layers
  n_hidden_layers_y = hyperparams$n_hidden_layers_y; n_hidden_layers_r = hyperparams$n_hidden_layers_r; dim_zs = hyperparams$dim_zs
  L1_weights = hyperparams$L1_weights
  #####################
  
  # (family, link) = (Gaussian, identity), (Multinomial, mlogit), (Poisson, log)

  np = reticulate::import("numpy")
  torch = reticulate::import("torch")
  # reticulate::source_python(system.file("dlglm.py", package = "dlglm"))   # once package is made, put .py in "inst" dir
  reticulate::source_python("dlglm.py")
  P = ncol(X); N = nrow(X)
  
  # Transform count data (log) and cat data (subtract by min)
  data_types_x_0 = data_types_x
  # X[,data_types_x=="count"] = log(X[,data_types_x=="count"]+0.001)
  if(sum(data_types_x=="cat") == 0){
    X_aug = X
    mask_x_aug = mask_x
    Cs = np$empty(shape=c(0L,0L))
  } else{
    # reorder to real&count covariates first, then augment cat dummy vars
    X_aug = X[,!(data_types_x %in% c("cat"))]
    mask_x_aug = mask_x[,!(data_types_x %in% c("cat"))]
    covars_r_x_aug = covars_r_x[!(data_types_x %in% c("cat"))]
    
    ## onehot encode categorical variables
    # X_cats = X[,data_types_x=="cat"]
    Cs = rep(0,sum(data_types_x=="cat"))
    # X_cats_onehot = matrix(nrow=N,ncol=0)
    cat_ids = which(data_types_x=="cat")
    for(i in 1:length(cat_ids)){
      
      X_cat = as.numeric(as.factor(X[,cat_ids[i]]))-1
      Cs[i] = length(unique(X_cat))
      
      X_cat_onehot = matrix(ncol = Cs[i], nrow=length(X_cat))
      for(ii in 1:Cs[i]){
        X_cat_onehot[,ii] = (X_cat==ii-1)^2
      }
      # X_cats_onehot = cbind(X_cats_onehot, X_cat_onehot)
      X_aug = cbind(X_aug, X_cat_onehot)
      mask_x_aug = cbind(mask_x_aug, matrix(mask_x[,cat_ids[i]], nrow=N, ncol=Cs[i]))
      covars_r_x_aug = c(covars_r_x_aug, rep(covars_r_x[data_types_x == "cat"][i], Cs[i]) )
    }
    
    
    ## column bind real/count and one-hot encoded cat vars
    data_types_x = c( data_types_x[!(data_types_x %in% c("cat"))], rep("cat",sum(Cs)) )
    covars_r_x = covars_r_x_aug
    Cs = np$array(Cs)
  }
  
  # X[,data_types=="cat"] = X[,data_types=="cat"] - apply(X[,data_types=="cat"],2,min)
  
  if(family=="Multinomial"){
    Y = Y-min(Y)
    # Y_aug = matrix(0,nrow=length(Y), ncol=length(unique(Y)))
    # for(i in 1:length(Y)){
    #   Y_aug[i,Y[i]+1] = 1 
    # }
    # mask_y_aug = matrix(mask_y,nrow=length(mask_y),ncol=2)
    Y_aug = Y
    mask_y_aug = mask_y
  }else{
    Y_aug = Y
    mask_y_aug = mask_y
  }   # set to start from 0, not 1
  # Xs = split(data.frame(X), g)        # split by $train, $test, and $valid
  Xs = split(data.frame(X_aug), g)        # split by $train, $test, and $valid
  # Ys = split(data.frame(Y), g)        # split by $train, $test, and $valid
  Ys = split(data.frame(Y_aug), g)        # split by $train, $test, and $valid
  Rxs = split(data.frame(mask_x_aug), g)
  # Rys = split(data.frame(mask_y), g)
  Rys = split(data.frame(mask_y_aug), g)
  
  # misc fixed params:
  # draw_miss = T
  pre_impute_value = 0
  # n_hidden_layers_r=0  # no hidden layers in decoder_r, no nodes in that hidden layer (doesn't matter what h3 is)
  phi0=NULL; phi=NULL # only input when using logistic regression (known coefs)
  
  if(normalize){
    norm_means_x=colMeans(Xs$train, na.rm=T); norm_sds_x=apply(Xs$train,2,function(y) sd(y,na.rm=T))
    norm_mean_y=0; norm_sd_y=1
    norm_means_x[data_types_x=="cat"] = 0; norm_sds_x[data_types_x=="cat"] = 1
    # if(family=="Multinomial"){
    #   norm_mean_y=0; norm_sd_y=1
    # } else{
    #   norm_mean_y=colMeans(Ys$train, na.rm=T); norm_sd_y=apply(Ys$train,2,function(y) sd(y,na.rm=T))
    # }
    dir_name = sprintf("%s/with_normalization",dir_name)
    ifelse(!dir.exists(dir_name),dir.create(dir_name),F)
  }else{
    P_aug = ncol(Xs$train)
    norm_means_x = rep(0, P_aug); norm_sds_x = rep(1, P_aug)
    norm_mean_y = 0; norm_sd_y = 1
  }
  
  
  if(sum(data_types_x=="cat") == 0){
    LBs_trainVal = matrix(nrow = length(hs)*length(bss)*length(lrs)*length(niws)*length(n_epochss)*length(n_hidden_layers)*length(n_hidden_layers_y)*length(n_hidden_layers_r)*length(dim_zs)*length(L1_weights),
                          ncol=13) #ncol = 13)
    colnames(LBs_trainVal) = c("h","bs","lr","niw","epochs","nhls","nhl_y","nhl_r","dim_z", "L1_weight",
                               "LB_train",#"MSE_train_x","MSE_train_y",
                               "LB_valid",#,"MSE_valid_x","MSE_valid_y"
                               "MSE_real"
    )
  }else{
    LBs_trainVal = matrix(nrow = length(hs)*length(bss)*length(lrs)*length(niws)*length(n_epochss)*length(n_hidden_layers)*length(n_hidden_layers_y)*length(n_hidden_layers_r)*length(dim_zs)*length(L1_weights),
                          ncol=15) #ncol = 13)
    colnames(LBs_trainVal) = c("h","bs","lr","niw","epochs","nhls","nhl_y","nhl_r","dim_z", "L1_weight",
                               "LB_train",#"MSE_train_x","MSE_train_y",
                               "LB_valid",#,"MSE_valid_x","MSE_valid_y"
                               "MSE_real","MSE_cat","PA_cat"
                               )
  }
  index = 1
  
  # i=1;j=1;k=1;m=1;mm=1;nn=1;oo=1
  print("data_types_x"); print(data_types_x)
  print("data_types_x_0"); print(data_types_x_0)
  torch$cuda$empty_cache()
  
  for(i in 1:length(hs)){for(j in 1:length(bss)){for(k in 1:length(lrs)){
    for(m in 1:length(niws)){for(mm in 1:length(n_epochss)){for(nn in 1:length(n_hidden_layers)){for(ny in 1:length(n_hidden_layers_y)){for(nr in 1:length(n_hidden_layers_r)){for(oo in 1:length(dim_zs)){for(pp in 1:length(L1_weights)){
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
                        learn_r, np$array(data_types_x), np$array(data_types_x_0), Cs, 
                        early_stop, np$array(Xs$valid), np$array(Rxs$valid), np$array(Ys$valid), np$array(Rys$valid),  ########## MIGHT NOT NEED Xs_val...--> may just take Xs$valid
                        Ignorable, family, link,
                        impute_bs, arch, draw_miss, 
                        pre_impute_value, n_hidden_layers[nn], n_hidden_layers_y[ny], n_hidden_layers_r[nr], 
                        hs[i], hs[i], hs[i], phi0, phi, 
                        1, NULL, sigma, bss[j], n_epochss[mm],
                        lrs[k], niws[m], 1L, dim_zs[oo], dir_name=dir_name, trace=trace, save_imps=F, test_temp=0.5, L1_weight=L1_weights[pp])   
      res_valid = dlglm(np$array(Xs$valid), np$array(Rxs$valid), np$array(Ys$valid), np$array(Rys$valid),
                        np$array(covars_r_x), np$array(covars_r_y),
                        np$array(norm_means_x), np$array(norm_sds_x), np$array(norm_mean_y), np$array(norm_sd_y),
                        learn_r, np$array(data_types_x), np$array(data_types_x_0), Cs,
                        F, NA, NA, NA, NA,
                        Ignorable, family, link,
                        impute_bs, arch, draw_miss, 
                        pre_impute_value, n_hidden_layers[nn], n_hidden_layers_y[ny], n_hidden_layers_r[nr], 
                        hs[i], hs[i], hs[i], phi0, phi, 
                        0, res_train$saved_model, sigma, bss[j], 2L,
                        lrs[k], niws[m], 1L, dim_zs[oo], dir_name=dir_name, trace=trace, save_imps=F, test_temp=res_train$'train_params'$'temp', L1_weight=res_train$'train_params'$'L1_weight')  # no early stopping in validation
      
      val_LB = res_valid$'LB'    # res_train$'val_LB'
      
      print(c(hs[i],bss[j],lrs[k],niws[m],n_epochss[mm],
              n_hidden_layers[nn],n_hidden_layers_y[ny],n_hidden_layers_r[nr],dim_zs[oo],L1_weights[pp],
              res_train$'LB', #res_train$'MSE'$miss_x[length(res_train$'MSE'$miss_x)],
              #res_train$'MSE'$miss_y[length(res_train$'MSE'$miss_y)],
              val_LB#, res_valid$'MSE'$miss_x[length(res_valid$'MSE'$miss_x)],
              #res_valid$'MSE'$miss_y[length(res_valid$'MSE'$miss_y)]
      ))
      print(res_valid$'errs')
      if(sum(data_types_x=="cat") == 0){
        LBs_trainVal[index,]=c(hs[i],bss[j],lrs[k],niws[m],n_epochss[mm],
                               n_hidden_layers[nn],n_hidden_layers_y[ny],n_hidden_layers_r[nr],dim_zs[oo],L1_weights[pp],
                               res_train$'LB', #res_train$'MSE'$miss_x[length(res_train$'MSE'$miss_x)],
                               #res_train$'MSE'$miss_y[length(res_train$'MSE'$miss_y)],
                               val_LB,
                               # res_train$'val_LB', #res_valid$'MSE'$miss_x[length(res_valid$'MSE'$miss_x)],
                               #res_valid$'MSE'$miss_y[length(res_valid$'MSE'$miss_y)]
                               res_valid$'errs'$'real'$'miss'
        )
      }else{
        LBs_trainVal[index,]=c(hs[i],bss[j],lrs[k],niws[m],n_epochss[mm],
                               n_hidden_layers[nn],n_hidden_layers_y[ny],n_hidden_layers_r[nr],dim_zs[oo],L1_weights[pp],
                               res_train$'LB', #res_train$'MSE'$miss_x[length(res_train$'MSE'$miss_x)],
                               #res_train$'MSE'$miss_y[length(res_train$'MSE'$miss_y)],
                               val_LB,
                               # res_train$'val_LB', #res_valid$'MSE'$miss_x[length(res_valid$'MSE'$miss_x)],
                               #res_valid$'MSE'$miss_y[length(res_valid$'MSE'$miss_y)]
                               res_valid$'errs'$'real'$'miss', res_valid$'errs'$'cat0'$'miss', res_valid$'errs'$'cat1'$'miss'
                               )
      }
      
      print(LBs_trainVal)
      if(is.na(val_LB)){val_LB=-Inf}
      
      # save only the best result currently (not all results) --> save memory
      if(index==1){opt_LB = val_LB; save(res_train, file=sprintf("%s/temp_opt_train.out",dir_name)); torch$save(res_train$'saved_model',sprintf("%s/temp_opt_train_saved_model.pth",dir_name))  #; save(opt_train, file="temp_opt_train.out")
      }else if(val_LB > opt_LB){opt_LB = val_LB; save(res_train, file=sprintf("%s/temp_opt_train.out",dir_name)); torch$save(res_train$'saved_model',sprintf("%s/temp_opt_train_saved_model.pth",dir_name))} #; save(opt_train, file="temp_opt_train.out")
      
      rm(res_train)
      rm(res_valid)
      
      index=index+1
      
      # release gpu memory
      # reticulate::py_run_string("import torch")
      # reticulate::py_run_string("torch.cuda.empty_cache()")
      torch$cuda$empty_cache()
      gc()
    }}}}}}}}}}
  
  saved_model = torch$load(sprintf("%s/temp_opt_train_saved_model.pth",dir_name))
  load(sprintf("%s/temp_opt_train.out",dir_name))
  train_params=res_train$train_params
  
  test_bs = 500L
  res_test = dlglm(np$array(Xs$test), np$array(Rxs$test), np$array(Ys$test), np$array(Rys$test),
                   np$array(covars_r_x), np$array(covars_r_y),
                   np$array(norm_means_x), np$array(norm_sds_x), np$array(norm_mean_y), np$array(norm_sd_y),
                   learn_r, np$array(data_types_x), np$array(data_types_x_0), Cs,
                   F, NA, NA, NA, NA, Ignorable, family, link,
                   test_bs, arch, draw_miss, 
                   train_params$pre_impute_value, train_params$n_hidden_layers, train_params$n_hidden_layers_y, train_params$n_hidden_layers_r, 
                   train_params$h1, train_params$h2, train_params$h3, phi0, phi, 
                   0, saved_model, sigma, test_bs, 2L,
                   train_params$lr, n_imps, 1L, train_params$dim_z, dir_name=dir_name, trace=trace, save_imps=T, test_temp=train_params$'temp', L1_weight=train_params$'L1_weight')
  
  fixed.params = list(dir_name=dir_name, covars_r_x=covars_r_x, covars_r_y=covars_r_y, learn_r=learn_r, Ignorable=Ignorable, family=family, link=link)
  
  return(list(results=res_test, fixed.params=fixed.params))
  
}

