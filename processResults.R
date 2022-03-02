processResults = function(prefix="",data.file.name = NULL, mask.file.name=NULL,
                          sim.params = list(N=1e5, D=2, P=8, data_types=NA, family="Gaussian", sim_index=1, ratios=c(train=.6,valid=.2,test=.2), mu=0, sd=1, beta=5, C=3, Cy=NULL, NL_x=F, NL_y=F),
                          miss.params = list(scheme="UV", mechanism="MNAR", pi=0.5, phi0=5, miss_pct_features=50, miss_cols=NULL, ref_cols=NULL, NL_r=F),
                          case=c("x","y","xy"), normalize=F, data_types_x, data_type_y = "real", methods=c("idlglm","dlglm","mice","zero","mean")){
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(gtable)
  library(mice)
  library(mclust)
  require(foreign)
  require(nnet)
  
  inv_logit = function(x){
    return(1/(1+exp(-x)))
  }  # can take this out once dlglm is packaged
  logsumexp <- function (x) {
    y = max(x)
    y + log(sum(exp(x - y)))
  }
  softmax <- function (x) {
    exp(x - logsumexp(x))
  }
  N=sim.params$N; P=sim.params$P; data_types=sim.params$data_types; family=sim.params$family
  link=if(family=="Gaussian"){"identity"}else if(family=="Multinomial"){"mlogit"}else if(family=="Poisson"){"log"}
  if(is.null(data.file.name)){
    dataset = sprintf("SIM_N%d_P%d_D%d", N, P, D)
  } else{dataset = unlist(strsplit(data.file.name,"[.]"))[1]}
  pi = miss.params$pi
  mechanism=miss.params$mechanism
  sim_index=sim.params$sim_index
  
  
  dir_name0 = sprintf("Results_X%s_Y%s/%s%s/miss_%s/phi%d/sim%d", data_type_x,data_type_y,prefix, dataset, case, miss.params$phi0, sim_index) 
  dir_name=dir_name0
  # to save interim results
  # if(Ignorable){dir_name = sprintf("%s/Ignorable",dir_name0)}else{dir_name=dir_name0}
  ifelse(!dir.exists(sprintf("%s/Diagnostics",dir_name)), dir.create(sprintf("%s/Diagnostics",dir_name)), F)
  ifelse(!dir.exists(sprintf("%s/%s_%d_%d",dir_name, mechanism, miss.params$miss_pct_features, pi*100)),
         dir.create(sprintf("%s/%s_%d_%d",dir_name, mechanism, miss.params$miss_pct_features, pi*100)), F)
  
  data.fname = sprintf("%s/data_%s_%d_%d.RData", dir_name0, mechanism, miss_pct_features, pi*100)
  print(paste("Data file: ", data.fname))
  if(!file.exists(data.fname)){
    stop("Data file does not exist..")
  } else{
    load(data.fname)
  }
  
  data_types_x_0 = data_types_x
  # mask_x = (res$mask_x)^2; mask_y = (res$mask_y)^2
  if(sum(data_types_x=="cat") == 0){
    X_aug = X
    # mask_x_aug = mask_x
  } else{
    # reorder to real&count covariates first, then augment cat dummy vars
    X_aug = X[,!(data_types_x %in% c("cat"))]
    # mask_x_aug = mask_x[,!(data_types_x %in% c("cat"))]
    
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
      # mask_x_aug = cbind(mask_x_aug, matrix(mask_x[,cat_ids[i]], nrow=N, ncol=Cs[i]))
    }
    
    
    ## column bind real/count and one-hot encoded cat vars
    data_types_x = c( data_types_x[!(data_types_x %in% c("cat"))], rep("cat",sum(Cs)) )
  }
  
  Xs = split(data.frame(X), g)        # split by $train, $test, and $valid
  Xs_aug = split(data.frame(X_aug), g)        # split by $train, $test, and $valid
  Ys = split(data.frame(Y), g)        # split by $train, $test, and $valid
  Rxs = split(data.frame(mask_x), g)
  Rys = split(data.frame(mask_y), g)
  norm_means_x=colMeans(Xs$train, na.rm=T); norm_sds_x=apply(Xs$train,2,function(y) sd(y,na.rm=T))   # normalization already undone in results xhat
  # norm_mean_y=colMeans(Ys$train, na.rm=T); norm_sd_y=apply(Ys$train,2,function(y) sd(y,na.rm=T))
  norm_mean_y=0; norm_sd_y=1   # didn't normalize Y
  
  miss_cols = which(colMeans(mask_x)!=1)
  
  tab = matrix(nrow = P, ncol=0)
  xhats = list()
  yhats = matrix(ncol=0,nrow=nrow(Ys$test))
  prhats = list()
  ## Process mean/zero imputation of X and Y ##
  if("zero" %in% methods){
    X_zero = X; X_zero[mask_x==0]=0
    Y_zero = Y; Y_zero[mask_y==0]=0
    Xs_zero = split(data.frame(X_zero), g)
    Ys_zero = split(data.frame(Y_zero), g)
    xhat_zero = Xs_zero$test; yhat_zero = Ys_zero$test
    
    if(family=="Gaussian"){
      fit_zero = glm(Y_zero ~ 0 + . , data=cbind( Xs_zero$train, Ys_zero$train ))
      w_zero = c(fit_zero$coefficients)
    }else if(family=="Multinomial"){
      fit_zero = multinom(Y_zero ~ 0 + ., data=cbind( Xs_zero$train, Ys_zero$train ))
      w_zero = c(coefficients(fit_zero))
    }
    yhat_zero_pred = predict(fit_zero, newdata=cbind(Xs$test,Ys$test,row.names = NULL))
    yhats = cbind(yhats,yhat_zero_pred); colnames(yhats)[ncol(yhats)] = "zero"
    prhat_zero_pred = predict(fit_zero, newdata=cbind(Xs$test,Ys$test,row.names = NULL), type="probs")
    prhats$"zero" = prhat_zero_pred
    xhats$"zero" = xhat_zero
    
    tab = cbind(tab,w_zero); colnames(tab)[ncol(tab)] = "zero"
  }
  if("mean" %in% methods){
    X_mean = X; Y_mean = Y
    for(i in 1:ncol(X_mean)){
      X_mean[mask_x[,i]==0,i] = mean(X[mask_x[,i]==1,i])
    }
    Y_mean[mask_y==0] = mean(Y[mask_y==1])
    Xs_mean = split(data.frame(X_mean), g)
    Ys_mean = split(data.frame(Y_mean), g)
    xhat_mean = Xs_mean$test; yhat_mean = Ys_mean$test
    
    if(family=="Gaussian"){
      fit_mean = glm(Y_mean ~ 0 + . , data=cbind( Xs_mean$train ,Ys_mean$train ))
      w_mean = c(fit_mean$coefficients)
    }else if(family=="Multinomial"){
      fit_mean = multinom(as.factor(Y_mean) ~ 0+., data=cbind( Xs_mean$train ,Ys_mean$train ))
      w_mean = c(coefficients(fit_mean))
    }
    yhat_mean_pred = predict(fit_mean, newdata=cbind(Xs$test,Ys$test,row.names = NULL))
    yhats = cbind(yhats,yhat_mean_pred); colnames(yhats)[ncol(yhats)] = "mean"
    prhat_mean_pred = predict(fit_mean, newdata=cbind(Xs$test,Ys$test,row.names = NULL), type="probs")
    prhats$"mean" = prhat_mean_pred
    xhats$"mean" = xhat_mean
    
    tab = cbind(tab,w_mean); colnames(tab)[ncol(tab)] = "mean"
    
  }
  
  ## for idlglm, dlglm, mice --> load saved results
  if("dlglm" %in% methods){
    fname = sprintf("%s/res_dlglm_%s_%d_%d.RData",dir_name,mechanism,miss_pct_features,pi*100)
    print(paste("dlglm Results file: ", fname))
    if(!file.exists(fname)){ break }
    load( fname )  # loads "X","Y","mask_x","mask_y","g"
    # should contain "res", which is a list that contains "results" and "fixed.params" from now on. First iteration only contains res (no list) results object
    res = res$results
    niws = res$train_params$niws_z
    
    # Saving weights
    if(family=="Multinomial"){
      w0 = res$w0[-1] - res$w0[1] ## w0 should be nothing --> removed intercept to prevent multicollinearity
      w = res$w[-1,] - res$w[1,]    # reference is first class
      w_real = c(w[data_types_x=="real"])
      w_cat = c(w[data_types_x=="cat"])
    }else{
      w0 = res$w0 ## w0 should be nothing --> removed intercept to prevent multicollinearity
      w = res$w
      w_real = c(w[data_types_x=="real"])
      w_cat = c(w[data_types_x=="cat"])
    }
    
    if(family %in% c("Gaussian","Poisson")){
      mu_y = as.matrix(Xs_aug$test) %*% t(res$w) + as.matrix(rep(res$w0, nrow(Xs_aug$test)),ncol=1)
      mu_y2 = res$all_params$y$mean  # average over the multiple samples of Xm --> Y'1
    } else if(family=="Multinomial"){
      eta = as.matrix(Xs_aug$test) %*% t(res$w) + matrix(rep(res$w0, nrow(Xs_aug$test)),nrow=nrow(Xs_aug$test),ncol=nrow(res$w),byrow=T) 
      probs_y = t(apply(eta, 1, softmax))  # should this be softmax?
      mu_y = apply(probs_y,1,which.max)
      mu_y2 = apply(res$all_params$y$probs, 1,which.max)   # using the p(y|x) posterior mode of test set --> why? could just predict
    }
    yhats = cbind(yhats,mu_y); colnames(yhats)[ncol(yhats)] = "dlglm"
    # yhats = cbind(yhats,mu_y2); colnames(yhats)[ncol(yhats)] = "dlglm_mode"
    
    prhats$"dlglm" = probs_y[,-1]
    # prhats$"dlglm_mode" = res$all_params$y$probs[,-1]
    tab = cbind(tab,w); colnames(tab)[ncol(tab)] = "dlglm"
    
    xhats$"dlglm" = res$xhat

  }
  if("idlglm" %in% methods){
    fname = sprintf("%s/Ignorable/res_dlglm_%s_%d_%d.RData",dir_name,mechanism,miss_pct_features,pi*100)
    print(paste("idlglm Results file: ", fname))
    if(!file.exists(fname)){ break }
    load( fname )  # loads "X","Y","mask_x","mask_y","g"
    # should contain "res", which is a list that contains "results" and "fixed.params" from now on. First iteration only contains res (no list) results object
    ires = res$results
    iniws = ires$train_params$niws_z
    
    # Saving weights
    if(family=="Multinomial"){
      iw0 = ires$w0[-1] - ires$w0[1] ## w0 should be nothing --> removed intercept to prevent multicollinearity
      iw = ires$w[-1,] - ires$w[1,]    # reference is first class
      iw_real = c(iw[data_types_x=="real"])
      iw_cat = c(iw[data_types_x=="cat"])
    }else{
      iw0 = ires$w0 ## w0 should be nothing --> removed intercept to prevent multicollinearity
      iw = ires$w
      iw_real = c(iw[data_types_x=="real"])
      iw_cat = c(iw[data_types_x=="cat"])
    }
    
    if(family %in% c("Gaussian","Poisson")){
      imu_y = as.matrix(Xs_aug$test) %*% t(ires$w) + as.matrix(rep(ires$w0, nrow(Xs_aug$test)),ncol=1)
      imu_y2 = ires$all_params$y$mean  # average over the multiple samples of Xm --> Y'1
    } else if(family=="Multinomial"){
      ieta = as.matrix(Xs_aug$test) %*% t(ires$w) + matrix(rep(ires$w0, nrow(Xs_aug$test)),nrow=nrow(Xs_aug$test),ncol=nrow(ires$w),byrow=T) 
      iprobs_y = t(apply(ieta, 1, softmax))  # should this be softmax?
      imu_y = apply(iprobs_y,1,which.max)
      imu_y2 = apply(ires$all_params$y$probs, 1,which.max)   # using the p(y|x) posterior mode of test set --> why? could just predict
    }
    yhats = cbind(yhats,imu_y); colnames(yhats)[ncol(yhats)] = "idlglm"
    # yhats = cbind(yhats,imu_y2); colnames(yhats)[ncol(yhats)] = "idlglm_mode"
    
    prhats$"idlglm" = iprobs_y[,-1]
    # prhats$"idlglm_mode" = ires$all_params$y$probs[,-1]
    
    tab = cbind(tab,iw); colnames(tab)[ncol(tab)] = "idlglm"
    
    xhats$"idlglm" = ires$xhat
  }
  rm(Xs_aug)  # just need for dlglm
  
  
  if("mice" %in% methods){
    fname_mice = sprintf("%s/res_mice_%s_%d_%d.RData",dir_name0,mechanism,miss_pct_features,pi*100)
    print(paste("mice Results file: ", fname_mice))
    if(!file.exists(fname_mice)){ break }
    
    load( fname_mice )
    xhat_mice = res_mice$xhat_mice; xyhat_mice = res_mice$xyhat_mice  # training x/yhats
    res_MICE = res_mice$res_MICE; res_MICE_test=res_mice$res_MICE_test
    rm(res_mice)
    
    if(family=="Gaussian"){
      fits_MICE = list()
      for(i in 1:res_MICE$m){
        fits_MICE[[i]] = glm(y ~ 0+., data=complete(res_MICE,i))
        fits_MICE[[i]]$data = NULL; fits_MICE[[i]]$model=NULL
      }
      fit_MICE = pool(fits_MICE)
    }else if(family=="Multinomial"){
      fits_MICE = list()
      for(i in 1:res_MICE$m){
        fits_MICE[[i]] = multinom(y ~ 0+., data=complete(res_MICE,i))
        fits_MICE[[i]]$data = NULL; fits_MICE[[i]]$model=NULL
      }
      fit_MICE = pool(fits_MICE)
    }
    
    dummy_fit_MICE = fits_MICE[[1]]; rm(fits_MICE)
    dummy_fit_MICE$coefficients = fit_MICE$pooled$estimate
    yhat_mice_pred = predict(dummy_fit_MICE,newdata = Xs$test)
    
    w_mice = c(fit_MICE$pooled$estimate)
    
    yhats = cbind(yhats,yhat_mice_pred); colnames(yhats)[ncol(yhats)] = "mice"
    prhat_mice_pred = predict(dummy_fit_MICE, newdata=Xs$test, type="probs")
    prhats$"mice" = prhat_mice_pred
    tab = cbind(tab,w_mice); colnames(tab)[ncol(tab)] = "mice"
    
    xhat_mice_test = mice::complete(res_MICE_test,1)
    for(c in 2:res_MICE_test$m){
      xhat_mice_test = xhat_mice_test + mice::complete(res_MICE_test,c)
    }
    xhat_mice_test = xhat_mice_test/res_MICE_test$m
    xhat_mice_test = xhat_mice_test[,-ncol(xhat_mice_test)]
    
    xhats$"mice" = xhat_mice
    
  }

  # load simulated parameters file
  load( sprintf("%s/params_%s_%d_%d.RData", dir_name0, mechanism, miss_pct_features,pi*100) )
  if(family=="Multinomial"){
    beta0s = sim.data$params$beta0s; betas = sim.data$params$betas[,-1] - sim.data$params$betas[,1]
    prs = split(data.frame(sim.data$params$prs[,-1]), g)   # first column: reference column.
    # prhats$"truth" = prs$test
  }else{ beta0s = sim.data$params$beta0s; betas = sim.data$params$betas }
  tab = cbind(betas,tab)
  
  if(family=="Gaussian"){
    true_mu_y = sim.data$params$beta0s + as.matrix(Xs$test) %*% as.matrix(sim.data$params$betas,ncol=1)
    # boxplot(true_mu_y - mu_y, main = "Difference between true E[Y|X] and predicted Y",outline=F)  # prediction of Y (mu_y as)
  }
  
  ## some dlglm-specific plots ##
  plots_dlglm = function(Ys_test, mu_y, family, dir_name, mechanism, miss_pct_features, pi, 
                         betas, w_real, w_cat, data_types_x, data_types_x_0, Ignorable){
    
    if(family%in% c("Gaussian","Poisson")){
      if(Ignorable){ fname = sprintf("%s/%s_%d_%d/pred_Y_idlglm.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100)
      }else{ fname = sprintf("%s/%s_%d_%d/pred_Y_dlglm.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100) }
      
      png(fname)
      boxplot(as.numeric(unlist(Ys$test - as.numeric(mu_y))), main="Diff between test and pred Y",outline=F)  # prediction of Y (mu_y as)
      dev.off()
    } else if(family=="Multinomial"){
      tab = table(mu_y,unlist(Ys_test))
      p<-tableGrob(round(tab,3))
      
      library(mclust)
      ARI_val = adjustedRandIndex(mu_y,unlist(Ys_test))
      title <- textGrob(sprintf("Y, Pred(r), true(c), ARI=%f",round(ARI_val,3)),gp=gpar(fontsize=8))
      padding <- unit(5,"mm")
      
      p <- gtable_add_rows(
        p, 
        heights = grobHeight(title) + padding,
        pos = 0)
      p <- gtable_add_grob(
        p, 
        title, 
        1, 1, 1, ncol(p))
      
      if(Ignorable){ fname = sprintf("%s/%s_%d_%d/pred_classes_idlglm.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100)
      }else{ fname = sprintf("%s/%s_%d_%d/pred_classes_dlglm.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100) }
      ggsave(p,file=fname, width=6, height=6, units="in")
    }
    
    fname = if(Ignorable){sprintf("%s/%s_%d_%d/coefs_real_idlglm.png",dir_name,mechanism, miss.params$miss_pct_features, pi*100)
      }else{sprintf("%s/%s_%d_%d/coefs_real_dlglm.png",dir_name,mechanism, miss.params$miss_pct_features, pi*100)}
    png(fname,res = 300,width = 4, height = 4, units = 'in')
    ymin = min(c(betas[data_types_x_0=="real"], w_real))
    ymax = max(c(betas[data_types_x_0=="real"], w_real))
    plot(c(betas[data_types_x_0=="real"]), main="True (black) vs fitted (red) real coefs", xlab="covariate", ylab="coefficient", ylim=c(ymin,ymax)); points(c(w_real),col="red", cex=0.5)
    dev.off()
    if(sum(data_types_x=="cat")>0){
      fname = if(Ignorable){sprintf("%s/%s_%d_%d/coefs_cat_idlglm.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100)
        }else{sprintf("%s/%s_%d_%d/coefs_cat_dlglm.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100)}
      png(fname,res = 300,width = 4, height = 4, units = 'in')
      ymin = min(c(betas[data_types_x_0=="cat"],sapply(split(w_cat, rep(1:length(Cs), Cs)), function(x){mean((x - c(0,x[-length(x)]))[-1])})))-0.005
      ymax = max(c(betas[data_types_x_0=="cat"],sapply(split(w_cat, rep(1:length(Cs), Cs)), function(x){mean((x - c(0,x[-length(x)]))[-1])})))+0.005
      plot(c(betas[data_types_x_0=="cat"]), main="True (black) vs fitted (red) cat coefs", xlab="covariate", ylab="coefficient", ylim=c(ymin,ymax))
      points(sapply(split(w_cat, rep(1:length(Cs), Cs)), function(x){mean((x - c(0,x[-length(x)]))[-1])}),col="red", cex=0.5)
      dev.off()
    }
  }
  
  if("dlglm" %in% methods){ plots_dlglm(Ys$test, mu_y, family, dir_name, mechanism, miss.params$miss_pct_features, pi, 
                                        betas, w_real, w_cat, data_types_x, data_types_x_0, F) }
  if("idlglm" %in% methods){ plots_dlglm(Ys$test, imu_y, family, dir_name, mechanism, miss.params$miss_pct_features, pi,
                                         betas, iw_real, iw_cat, data_types_x, data_types_x_0, T) }
  ###############################
  
  plots_others = function(Ys_test, yhat, family, dir_name, mechanism, miss_pct_features, pi,
                          betas, w, data_types_x_0, method){
    png(sprintf("%s/%s_%d_%d/coefs_real_%s.png",dir_name,mechanism, miss.params$miss_pct_features, pi*100, method),
        res = 300,width = 4, height = 4, units = 'in')
    ymin = min(c(betas[data_types_x_0=="real"], w))
    ymax = max(c(betas[data_types_x_0=="real"], w))
    plot(c(betas[data_types_x_0=="real"]), main="True (black) vs fitted (red) real coefs", xlab="covariate", ylab="coefficient", ylim=c(ymin,ymax)); points(c(w),col="red", cex=0.5)
    dev.off()
    
    if(family%in% c("Gaussian","Poisson")){
      fname = sprintf("%s/%s_%d_%d/pred_Y_%s.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100,method)
      png(fname)
      boxplot(as.numeric(unlist(Ys_test - as.numeric(yhat))), main="Diff between test and pred Y",outline=F)  # prediction of Y (mu_y as)
      dev.off()
    } else if(family=="Multinomial"){
      tab = table(yhat,unlist(Ys_test))
      p<-tableGrob(round(tab,3))
      
      library(mclust)
      ARI_val = adjustedRandIndex(yhat,unlist(Ys_test))
      title <- textGrob(sprintf("Y, Pred(r), true(c), ARI=%f",round(ARI_val,3)),gp=gpar(fontsize=8))
      padding <- unit(5,"mm")
      
      p <- gtable_add_rows(
        p, 
        heights = grobHeight(title) + padding,
        pos = 0)
      p <- gtable_add_grob(
        p, 
        title, 
        1, 1, 1, ncol(p))
      
      fname = sprintf("%s/%s_%d_%d/pred_classes_%s.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100,method)
      ggsave(p,file=fname, width=6, height=6, units="in")
    }
  }
  if("zero" %in% methods){ plots_others(Ys$test, yhat_zero_pred, family, dir_name, mechanism, miss.params$miss_pct_features, pi,
                                        betas, w_zero, data_types_x_0, "zero") }
  if("mean" %in% methods){ plots_others(Ys$test, yhat_mean_pred, family, dir_name, mechanism, miss.params$miss_pct_features, pi,
                                        betas, w_mean, data_types_x_0, "mean") }
  if("mice" %in% methods){ plots_others(Ys$test, yhat_mice_pred, family, dir_name, mechanism, miss.params$miss_pct_features, pi,
                                       betas, w_mice, data_types_x_0, "mice") }
  
  
  # comparing all
  L1s_y = abs( matrix(unlist(Ys$test),nrow=nrow(Ys$test),ncol=ncol(yhats)) - yhats )
  if(family%in% c("Gaussian","Poisson")){
    png(filename=sprintf("%s/%s_%d_%d/predY.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 6, height = 6, units = 'in')
    boxplot(L1s_y,names=colnames(L1s_y), outline=F,
            main="Absolute Difference between test Y and predicted Y")
    # boxplot(as.numeric(unlist(abs(Ys$test - mu_y))),
    #         as.numeric(unlist(abs(Ys$test - as.numeric(yhat_zero_pred)))),
    #         names=c("dlglm","zero"), outline=F,
    #         main="Absolute Difference between test Y and predicted Y")
    dev.off()
    
  }
  
  if(family=="Multinomial"){
    p = tableGrob(round(t(unlist(lapply(prhats,function(x){mean(abs(x - c(prs$test)[[1]]))}))), 3))
    title <- textGrob("Mean Diff Between True vs Pred Probs_y",gp=gpar(fontsize=9))
    padding <- unit(5,"mm")
    
    p <- gtable_add_rows(
      p, 
      heights = grobHeight(title) + padding,
      pos = 0)
    p <- gtable_add_grob(
      p, 
      title, 
      1, 1, 1, ncol(p))
    ggsave(p,file=sprintf("%s/%s_%d_%d/mean_L1_probY.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100), width=6, height=4, units="in")
    
    prs_L1 = lapply(prhats, function(x){abs(x - c(prs$test)[[1]])})
    png(filename=sprintf("%s/%s_%d_%d/L1s_probY.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 10, height = 10, units = 'in')
    boxplot(prs_L1, names=names(prs_L1),
            outline=F, main="Abs deviation between true and pred Pr(Y)")
    dev.off()
  }
  
  
  ################### TOGETHER PLOTS
  
  p<-tableGrob(round(tab,3))
  title <- textGrob("Wts from dlglm/idlglm, other coefs imp + glm (no int)",gp=gpar(fontsize=9))
  padding <- unit(5,"mm")
  
  p <- gtable_add_rows(
    p, 
    heights = grobHeight(title) + padding,
    pos = 0)
  p <- gtable_add_grob(
    p, 
    title, 
    1, 1, 1, ncol(p))
  ggsave(p,file=sprintf("%s/%s_%d_%d/coefs_tab.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100), width=6, height=25, units="in")
  
  
  ## Compute RB, PB, AW, CR
  RB = tab[,-1] - tab[,1]
  PB = 100*abs(tab[,-1] - tab[,1])/abs(tab[,1] + 1e-10)  # PB will blow up to a very large number if true coef est is 0
  SE = sqrt((tab[,-1] - tab[,1])^2)   # sqrt/mean across sim index to get RMSE
  
  df = cbind(RB, PB, SE)
  df = rbind(df, c( colMeans(RB), colMeans(PB), colMeans(SE) ))
  rownames(df) = c(1:length(c(betas)), "Averaged")
  
  colnames(tab)[-1]
  colnames(df) = paste(rep(c("RB","PB","SE"),each=ncol(tab)-1),
                       colnames(tab)[-1],sep="_")
  
  p<-tableGrob(round(df,3))
  title <- textGrob("Raw and Percent Bias (RB/PB), and Squared error (SE) of dlglm (dl), mean (me), zero (ze), and mice (mi)",gp=gpar(fontsize=9))
  padding <- unit(5,"mm")
  
  p <- gtable_add_rows(
    p, 
    heights = grobHeight(title) + padding,
    pos = 0)
  p <- gtable_add_grob(
    p, 
    title, 
    1, 1, 1, ncol(p))
  ggsave(p,file=sprintf("%s/%s_%d_%d/coefs_tab2.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100), width=25, height=floor(P/2), units="in")
  
  
  ### CANT DO CR WITH DLGLM ###
  # summ_dlglm = cbind(tab[,2], NA,NA,NA)  # no CI for dlglm
  # summ_mean = cbind(summary(fit_mean)$coefficients, summary(fit_mean)$standard.errors, confint(fit_mean)); colnames(summ_mean) = c("estimate", "std.error", "2.5 %", "97.5 %")
  # summ_zero = cbind(summary(fit_zero)$coefficients, summary(fit_zero)$standard.errors, confint(fit_zero)); colnames(summ_zero) = c("estimate", "std.error", "2.5 %", "97.5 %")
  # summ_mice = summary(fit_MICE, "all", conf.int = TRUE)[, c("estimate", "std.error", "2.5 %", "97.5 %")]
  # coverage_width = function(summ, truth){
  #   # summ: contains 4 columns: estimate, std.error, 2.5 %, and 97.5 %
  #   # truth is just the true values
  #    coverage = (summ[,"2.5 %"] < truth & summ[,"97.5 %"] > truth)^2
  #    width = summ[,"97.5 %"] - summ[,"2.5 %"]
  # 
  #    return(list(coverage=coverage, width=width))
  # }
  # CR = cbind( coverage_width(summ_mean, tab[,1])$coverage,
  #              coverage_width(summ_zero, tab[,1])$coverage,
  #              coverage_width(summ_mice, tab[,1])$coverage)
  # AW = cbind( coverage_width(summ_mean, tab[,1])$width,
  #              coverage_width(summ_zero, tab[,1])$width,
  #              coverage_width(summ_mice, tab[,1])$width)
  # 
  # df = cbind(CR, AW)
  # df = rbind(df, c( colMeans(CR), colMeans(AW) ))
  # rownames(df) = c(1:length(c(betas)), "Averaged")
  # colnames(df) = c("CR_me", "CR_ze", "CR_mi",
  #                  "AW_me", "AW_ze", "AW_mi")
  # 
  # p<-tableGrob(round(df,3))
  # title <- textGrob("Coverage (CR) and Average width (AW) of CI's of estimates from mean (me), zero (ze), and mice (mi)",gp=gpar(fontsize=9))
  # padding <- unit(5,"mm")
  # 
  # p <- gtable_add_rows(
  #   p,
  #   heights = grobHeight(title) + padding,
  #   pos = 0)
  # p <- gtable_add_grob(
  #   p,
  #   title,
  #   1, 1, 1, ncol(p))
  # ggsave(p,file=sprintf("%s/%s_%d_%d/coefs_tab3.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100), width=12, height=15, units="in")

  
  
  rel_diffs = L1s_y/max(abs(Ys$test),0.001)
  pct_diffs = colMeans(rel_diffs)
  
  png(filename=sprintf("%s/%s_%d_%d/predY_rel.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 12, height = 12, units = 'in')
  boxplot(rel_diffs, outline=F,
          names=colnames(rel_diffs),
          main="Relative Difference between test Y and predicted Y")
  dev.off()

  p<-tableGrob(t(round(pct_diffs,3)))
  title <- textGrob("Rel diff between true and pred Ys",gp=gpar(fontsize=8.5))
  p <- gtable_add_rows(
    p, 
    heights = grobHeight(title) + padding,
    pos = 0)
  p <- gtable_add_grob(
    p, 
    title, 
    1, 1, 1, ncol(p))
  ggsave(p,file=sprintf("%s/%s_%d_%d/mean_pctdiff_predY.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100), width=6, height=4, units="in")

  ###########################################
  ###### look at imputation in missing ######
  ###########################################
  
  diffs = lapply(xhats, function(x) abs(x - Xs$test))
  L1s = lapply(diffs, function(x) x[Rxs$test==0])
  pcts = lapply(diffs, function(x) (x/min(abs(Xs$test), 0.001))[Rxs$test==0])
  # Mean/zero imputation results:
  
  # compare both imputation results
  if(grepl("x",case)){
    for(c in miss_cols){
      ids = Rxs$test[,c]==0
      df = lapply(diffs,function(x) x[ids,c])
      png(filename=sprintf("%s/%s_%d_%d/imputedX_col%d.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100,c),res = 300,width = 10, height = 10, units = 'in')
      boxplot(df, names=names(df),
              outline=F, main=sprintf("Abs dev between true and imputed X in col%d",c))
      dev.off()
    }
    png(filename=sprintf("%s/%s_%d_%d/imputedX.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 10, height = 10, units = 'in')
    boxplot(L1s, names=names(L1s),
            outline=F, main="Absolute deviation between true and imputed X")
    dev.off()
    png(filename=sprintf("%s/%s_%d_%d/imputedpctX.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 10, height = 10, units = 'in')
    boxplot(pcts, names=names(pcts),
            outline=F, main="Relative difference between true and imputed X")
    dev.off()
    
    
    # tiff(filename=sprintf("%s/%s_%d_%d/scatter_imputedX_refzero.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 5, height = 5, units = 'in')
    # plot(x=L1s_x-L1s_zero_x,y=L1s_mean_x-L1s_zero_x, main="Err dlglm vs mean imputed X, wrt zero-imputed X")
    # dev.off()
    # 
    # tiff(filename=sprintf("%s/%s_%d_%d/scatter_imputedX_refzero_mice.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 5, height = 5, units = 'in')
    # plot(x=L1s_mice_x-L1s_zero_x,y=L1s_mean_x-L1s_zero_x, main="Err mice vs mean imputed X, wrt zero-imputed X")
    # dev.off()
    
    tab3 = t(unlist(lapply(L1s,mean)))
    p<-tableGrob(round(tab3,3))
    title <- textGrob("Mean abs diff: true - imputed Xs",gp=gpar(fontsize=8.5))
    p <- gtable_add_rows(
      p, 
      heights = grobHeight(title) + padding,
      pos = 0)
    p <- gtable_add_grob(
      p, 
      title, 
      1, 1, 1, ncol(p))
    ggsave(p,file=sprintf("%s/%s_%d_%d/mean_L1s_X.png",dir_name, mechanism, miss.params$miss_pct_features, pi*100), width=6, height=4, units="in")
  }
  # if(grepl("y",case)){
  #   tiff(filename=sprintf("%s/%s_%d_%d/imputedY.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 5, height = 5, units = 'in')
  #   boxplot(L1s_y,L1s_mean_y,L1s_zero_y,
  #           names=c("dlglm","mean","zero"),
  #           outline=F, main="Absolute deviation between true and imputed Y")
  #   dev.off()
  #   tiff(filename=sprintf("%s/%s_%d_%d/imputedpctY.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 5, height = 5, units = 'in')
  #   boxplot(pcts_y,pcts_mean_y,pcts_zero_y,
  #           names=c("dlglm","mean","zero"),
  #           outline=F, main="Relative difference between true and imputed Y")
  #   dev.off()
  #   
  #   tiff(filename=sprintf("%s/%s_%d_%d/imputedY_refzero.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),res = 300,width = 5, height = 5, units = 'in')
  #   boxplot(L1s_y-L1s_zero_y,
  #           names=c("dlglm"),
  #           outline=F, main="Err imputed Y - Err zero-imputed Y")
  #   abline(h=L1s_mean_y[1]-L1s_zero_y[1], col = "Red")
  #   dev.off()
  # }
  
  ## TRYING: glm on dlglm resulting xhat (test data)
  # d_dlglm = cbind( xhat, Ys$test )
  # fit_dlglm = glm(Y ~ 0 + . , data=d_dlglm)
  
  # # mean - dlglm pairs plot
  # library(GGally)
  # my_fn = function(data, mapping, ...){
  #   p <- ggplot(data=data,mapping=mapping) + geom_point() + geom_abline(slope=1, intercept=0, colour="red")
  #   p
  # }
  # # for(c in miss_cols){
  #   # mask_ids = Rxs$test[,c]==0
  #   p = ggpairs(data.frame(Ys$test,
  #                            "dlglm"=as.numeric(mu_y),
  #                            "mean"=yhat_mean_pred,
  #                            "zero"=yhat_zero_pred,
  #                             "mice"=yhat_mice_pred), lower=list(continuous=my_fn),
  #               title="Plots of true vs predicted values of Y")
  #   ggsave(sprintf("%s/%s_%d_%d/pairs.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),p,height=6, width=6, units="in")
  #   df = data.frame("dlglm"=Ys$test-as.numeric(mu_y),
  #                   "mean"=Ys$test-as.numeric(yhat_mean_pred),
  #                   "zero"=Ys$test-as.numeric(yhat_zero_pred),
  #                   "mice"=Ys$test-as.numeric(yhat_mice_pred))
  #   names(df) = c("dlglm","mean","zero","mice")
  #   p = ggpairs(df,lower=list(continuous=my_fn),
  #               title="Plots of diffs between true vs pred values of Y")
  #   ggsave(sprintf("%s/%s_%d_%d/pairs_diff.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),p,height=6, width=6, units="in")
  #   df=data.frame("dlglm"=abs(Ys$test-as.numeric(mu_y)),
  #                 "mean"=abs(Ys$test-as.numeric(yhat_mean_pred)),
  #                 "zero"=abs(Ys$test-as.numeric(yhat_zero_pred)),
  #                 "mice"=abs(Ys$test-as.numeric(yhat_mice_pred)))
  #   names(df) = c("dlglm","mean","zero","mice")
  #   p = ggpairs(df,lower=list(continuous=my_fn),
  #               title="Plots of abs diffs between true vs pred values of Y")
  #   ggsave(sprintf("%s/%s_%d_%d/pairs_absdiff.tiff",dir_name, mechanism, miss.params$miss_pct_features, pi*100),p,height=6, width=6, units="in")
  # # }
  
}

# prefix="Xmean4/"
# prefix="Xmean10sd2_beta5_pi50/"
# mu=5; sd=5; beta=0.5; pi=0.5; miss_pct_features = 50
# mu=5; sd=5; beta=0.5; pi=0.3; miss_pct_features = 10
mu=1; sd=0; beta=0.25; pi=0.3; miss_pct_features = 50
# prefix=sprintf("Xmean%dsd%d_beta%d_pi%d_corr0.2/",mu,sd,beta,pi*100)
# N=1e5; P=8; phi0=100
# N=1e5; P=10; phi0=5
# N=1e5; P=10; D=2; phi0=1
# N=1e5; P=50; D=2; phi0=1
N=1e5; P=50; D=2; phi0=5
# N=1e5; P=100; D=2; phi0=1
data.file.name = NULL; mask.file.name=NULL
case="x"; normalize=F
# data_type_y = "real"; Cy=NULL   # real, cat, cts
data_type_y = "cat"; Cy=2   # real, cat, cts

# data_types_x = rep("real",8)
# data_types_x = c(rep("real",5), rep("cat",5)); C=3
data_types_x = rep("real",P); C=NULL
data_type_x = if(all(data_types_x==data_types_x[1])){data_types_x[1]}else{"mixed"}
P_real=sum(data_types_x=="real"); P_count=sum(data_types_x=="count"); P_cat=sum(data_types_x=="cat")
prefix=sprintf("Xr%dct%dcat%d_beta%f_pi%d/",P_real,P_count,P_cat,beta,pi*100)
sim_indexes = 1; mechanisms=c("MNAR","MAR","MCAR") #; Ignorables=c(F,T)
NL_x=F; NL_y=F; NL_r=F

methods=c("idlglm","dlglm","mice","zero","mean")

for(s in 1:length(sim_indexes)){for(m in 1:length(mechanisms)){
  #for(ii in 1:length(Ignorables)){
  
  sim_index=sim_indexes[s]; mechanism=mechanisms[m]
  sim.params = list(N=N, P=P, D=D, data_types=rep(data_type_x,P),
                    family=if(data_type_y=="real"){"Gaussian"}else if(data_type_y=="cat"){"Multinomial"}else if(data_typ_y=="cts"){"Poisson"},
                    sim_index=sim_index, ratios=c(train=.8,valid=.1,test=.1),
                    mu=mu, sd=sd, beta=beta, C=C, Cy=Cy, NL_x=NL_x, NL_y=NL_y)
  miss.params = list(scheme="UV", mechanism=mechanism, pi=pi, phi0=phi0, miss_pct_features=miss_pct_features,NL_r=F)
  
  processResults(prefix=prefix,data.file.name = data.file.name, mask.file.name=mask.file.name,
                 sim.params = sim.params,
                 miss.params = miss.params,
                 case=case, normalize=normalize,
                 data_types_x=data_types_x, data_type_y = data_type_y, methods=methods)
  
}}#}
