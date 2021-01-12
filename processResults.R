processResults = function(prefix="",data.file.name = NULL, mask.file.name=NULL,
                          sim.params = list(N=1e5, P=8, data_types=NA, family="Gaussian", sim_index=1, ratios=c(train=.6,valid=.2,test=.2), mu=0, sd=1, beta=5, C=3),
                          miss.params = list(scheme="UV", mechanism="MNAR", pi=0.5, phi0=5, miss_cols=NULL, ref_cols=NULL),
                          case=c("x","y","xy"), normalize=F, data_type_x="real", data_type_y = "real"){
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(gtable)
  
  N=sim.params$N; P=sim.params$P; data_types=sim.params$data_types; family=sim.params$family
  link=if(family=="Gaussian"){"identity"}else if(family=="Multinomial"){"mlogit"}else if(family=="Poisson"){"log"}
  if(is.null(data.file.name)){
    dataset = sprintf("SIM_N%d_P%d_X%s_Y%s", N, P, data_types[1], family)
  } else{dataset = unlist(strsplit(data.file.name,"[.]"))[1]}
  pi = miss.params$pi
  mechanism=miss.params$mechanism
  sim_index=sim.params$sim_index
  
  
  dir_name = sprintf("Results_X%s_Y%s/%s%s/miss_%s/phi%d/sim%d", data_type_x,data_type_y,prefix, dataset, case, miss.params$phi0, sim_index)   # to save interim results
  ifelse(!dir.exists(sprintf("%s/Diagnostics",dir_name)), dir.create(sprintf("%s/Diagnostics",dir_name)), F)
  
  data.fname = sprintf("%s/data_%s_%d.RData", dir_name, mechanism, pi*100)
  print(paste("Data file: ", data.fname))
  if(!file.exists(data.fname)){
    stop("Data file does not exist..")
  } else{
    load(data.fname)
  }
  
  fname = sprintf("%s/res_dlglm_%s_%d.RData",dir_name,mechanism,pi*100)
  print(paste("Results file: ", fname))
  if(!file.exists(fname)){
    stop("Results file does not exist..")
  } else{
    load( fname )  # loads "X","Y","mask_x","mask_y","g"
    # should contain "res", which is a list that contains "results" and "fixed.params" from now on. First iteration only contains res (no list) results object
    res = res$results
  }
  
  # mask_x = (res$mask_x)^2; mask_y = (res$mask_y)^2
  
  Xs = split(data.frame(X), g)        # split by $train, $test, and $valid
  Ys = split(data.frame(Y), g)        # split by $train, $test, and $valid
  Rxs = split(data.frame(mask_x), g)
  Rys = split(data.frame(mask_y), g)
  norm_means_x=colMeans(Xs$train, na.rm=T); norm_sds_x=apply(Xs$train,2,function(y) sd(y,na.rm=T))
  norm_mean_y=colMeans(Ys$train, na.rm=T); norm_sd_y=apply(Ys$train,2,function(y) sd(y,na.rm=T))
  
  miss_cols = which(colMeans(mask_x)!=1)
  
  str(res)
  
  niws=1
  if(grepl("x",case)){
    # NRMSE(X, res$xhat)
    niws=niws*res$train_params$L  # first iteration doesn't have this
  }
  if(grepl("y",case)){
    # NRMSE(Y, res$yhat)
    niws=niws*res$train_params$M
  }
  
  # just for first iteration:
  # niws=5
  
  ## Process mean/zero imputation of X and Y ##
  X_zero = X; X_zero[mask_x==0]=0
  Y_zero = Y; Y_zero[mask_y==0]=0
  X_mean = X; Y_mean = Y
  for(i in 1:ncol(X_mean)){
    X_mean[mask_x[,i]==0,i] = mean(X[mask_x[,i]==1,i])
  }
  Y_mean[mask_y==0] = mean(Y[mask_y==1])
  
  
  Xs_zero = split(data.frame(X_zero), g);  Xs_mean = split(data.frame(X_mean), g)
  Ys_zero = split(data.frame(Y_zero), g); Ys_mean = split(data.frame(Y_mean), g)

  
  xhat_zero = Xs_zero$test; yhat_zero = Ys_zero$test
  xhat_mean = Xs_mean$test; yhat_mean = Ys_mean$test
  
  #########################################
  ###### look at E[Y|X] in test data ######
  #########################################
  
  ## prediction of Y in test set: E[Y|X] - Ytrue
  if(normalize){
    mu_y = colMeans(matrix(res$all_params$y$mean*norm_sd_y + norm_mean_y,nrow=niws))  # average over the multiple samples of Xm --> Y'1
  }else{
    if(family %in% c("Gaussian","Poisson")){ mu_y = rowMeans(matrix(res$all_params$y$mean,ncol=niws))  # average over the multiple samples of Xm --> Y'1
    } else if(family=="Multinomial"){
      ## average probs, then choose max prob class
      # probs_y = apply(res$all_params$y$probs, 2, function(x){rowMeans(matrix(x,ncol=niws))})
      # mu_y = apply(probs_y,1,which.max)
      
      ## choose max classes first then take consensus
      cls_y = apply(res$all_params$y$probs, 1,which.max)
      mu_y = apply(matrix(cls_y, ncol=niws), 1, function(x){ as.numeric(names(which.max(table(x)))) })
    }
  }
  if(family%in% c("Gaussian","Poisson")){
    boxplot(as.numeric(unlist(Ys$test - mu_y)), main="Difference between test Y and learned E[Y|X]",outline=F)  # prediction of Y (mu_y as)
  } else if(family=="Multinomial"){
    tab = table(mu_y,unlist(Ys$test))
    p<-tableGrob(round(tab,3))
    
    library(mclust)
    ARI_val = adjustedRandIndex(mu_y,unlist(Ys$test))
    title <- textGrob(sprintf("Pred (r) x true (c), ARI=%f",ARI_val),gp=gpar(fontsize=9))
    padding <- unit(5,"mm")
    
    p <- gtable_add_rows(
      p, 
      heights = grobHeight(title) + padding,
      pos = 0)
    p <- gtable_add_grob(
      p, 
      title, 
      1, 1, 1, ncol(p))
    
    ggsave(p,file=sprintf("%s/Diagnostics/%s_pred_classes.png",dir_name,mechanism), width=6, height=6, units="in")
  }
  
  # load simulated parameters file
  load( sprintf("%s/params_%s_%d.RData", dir_name, mechanism, pi*100) )
  beta0s = sim.data$params$beta0s; betas = sim.data$params$betas
  if(family=="Gaussian"){
    true_mu_y = sim.data$params$beta0s + as.matrix(Xs$test) %*% as.matrix(sim.data$params$betas,ncol=1)
    boxplot(true_mu_y - mu_y, main = "Difference between true E[Y|X] and learned E[Y|X]",outline=F)  # prediction of Y (mu_y as)
  }
  
  # working with weights
  ## after first iteration: should ahve w0/w saved properly
  w0=res$w0
  w=res$w
  
  ## first iteration:
  # library(reticulate)
  # torch = import("torch")
  # saved_model = torch$load(sprintf("%s/%s_%d/temp_opt_train_saved_model.pth",dir_name,mechanism,pi*100))
  # # py_run_string("w0 = r.saved_model['NN_y'][0].bias.cpu().data.numpy()")
  # py_run_string("w = r.saved_model['NN_y'][0].weight.cpu().data.numpy()")
  # # w0=py$w0
  # w=py$w
  # print(paste("true intercept:",beta0s, ", fitted intercept:", w0))
  print("true, fitted coefs:")
  print(cbind(c(betas),c(w)))
  # w0 = saved_model$NN_y[0]$bias
  # w = saved_model$NN_y[0]$weight
  
  tiff(sprintf("%s/Diagnostics/%s_coefs.tiff",dir_name,mechanism),res = 300,width = 4, height = 4, units = 'in')
  plot(c(betas), main="True (black) vs fitted (red) coefs", xlab="covariate", ylab="coefficient"); points(c(w),col="red")
  dev.off()
  
  print(cbind(c(betas), c(w)))
  
  # Mean/zero imputation results:
  
  # d_mean = cbind( rbind(Xs_mean$train, Xs_mean$valid) ,rbind(Ys_mean$train, Ys_mean$valid) )
  # d_zero = cbind( rbind(Xs_zero$train, Xs_zero$valid), rbind(Ys_zero$train, Ys_zero$valid) )
  d_mean = cbind( Xs_mean$train ,Ys_mean$train )
  d_zero = cbind( Xs_zero$train, Ys_zero$train )
  
  # fit_mean = glm(Y_mean ~ . , data=d_mean)
  # fit_zero = glm(Y_zero ~ . , data=d_zero)
  if(family=="Gaussian"){
    fit_mean = glm(Y_mean ~ 0 + . , data=d_mean)
    fit_zero = glm(Y_zero ~ 0 + . , data=d_zero)
  }else if(family=="Multinomial"){
    require(foreign)
    require(nnet)
    fit_mean = multinom(as.factor(Y_mean) ~ 0+., data=d_mean)
    fit_zero = multinom(Y_zero ~ 0 + ., data=d_zero)
  }
  
  yhat_mean_pred = predict(fit_mean, newdata=cbind(Xs_mean$test,Ys_mean$test,row.names = NULL))
  yhat_zero_pred = predict(fit_zero, newdata=cbind(Xs_zero$test,Ys_zero$test,row.names = NULL))
  
  
  # comparing all
  if(family%in% c("Gaussian","Poisson")){
    boxplot(as.numeric(unlist(Ys$test - yhat_mean_pred)), main="Difference between test Y and predicted Y with mean-imputed X",outline=F)  # prediction of Y (mu_y as)
    boxplot(as.numeric(unlist(Ys$test - yhat_zero_pred)), main="Difference between test Y and predicted Y with zero-imputed X",outline=F)  # prediction of Y (mu_y as)
    
    tiff(filename=sprintf("%s/Diagnostics/%s_predY.tiff",dir_name,mechanism),res = 300,width = 6, height = 6, units = 'in')
    boxplot(as.numeric(unlist(abs(Ys$test - mu_y))),
            as.numeric(unlist(abs(Ys$test - yhat_mean_pred))),
            as.numeric(unlist(abs(Ys$test - yhat_zero_pred))),
            names=c("dlglm","mean","zero"), outline=F,
            main="Absolute Difference between test Y and predicted Y")
    # boxplot(as.numeric(unlist(abs(Ys$test - mu_y))),
    #         as.numeric(unlist(abs(Ys$test - yhat_zero_pred))),
    #         names=c("dlglm","zero"), outline=F,
    #         main="Absolute Difference between test Y and predicted Y")
    dev.off()
    
    L1_dlglm = as.numeric(unlist(abs(Ys$test - mu_y)))
    L1_mean = as.numeric(unlist(abs(Ys$test - yhat_mean_pred)))
    L1_zero = as.numeric(unlist(abs(Ys$test - yhat_zero_pred)))
    
    plot(x=L1_dlglm,y=L1_mean,main=c("Absolute Deviation between true and pred Y using mean+glm vs dlglm"))
    abline(0,1, col="red")
  }else if(family=="Multinomial"){
    tab1 = table(mu_y,unlist(yhat_mean_pred))
    tab2 = table(mu_y,yhat_zero_pred)
    p1<-tableGrob(round(tab1,3))
    p2<-tableGrob(round(tab2,3))
    
    ARI_val1 = adjustedRandIndex(mu_y,unlist(yhat_mean_pred))
    ARI_val2 = adjustedRandIndex(mu_y,unlist(yhat_zero_pred))
    title1 <- textGrob(sprintf("mean-Pred(r) x true(c),ARI=%f",ARI_val1),gp=gpar(fontsize=8))
    title2 <- textGrob(sprintf("zero-Pred(r) x true(c),ARI=%f",ARI_val2),gp=gpar(fontsize=8))
    padding <- unit(5,"mm")
    
    p1 <- gtable_add_rows(
      p1, 
      heights = grobHeight(title1) + padding,
      pos = 0)
    p1 <- gtable_add_grob(
      p1, 
      title1, 
      1, 1, 1, ncol(p1))
    p2 <- gtable_add_rows(
      p2, 
      heights = grobHeight(title2) + padding,
      pos = 0)
    p2 <- gtable_add_grob(
      p2, 
      title2, 
      1, 1, 1, ncol(p2))
    
    ggsave(p1,file=sprintf("%s/Diagnostics/%s_mean_pred_classes.png",dir_name,mechanism), width=6, height=6, units="in")
    ggsave(p2,file=sprintf("%s/Diagnostics/%s_zero_pred_classes.png",dir_name,mechanism), width=6, height=6, units="in")
  }
  if(family%in%c("Gaussian","Poisson")){
    # tab = cbind(c(beta0s,betas), c(w0,w), c(fit_mean$coefficients), c(fit_zero$coefficients))
    tab = cbind(c(betas), c(w), c(fit_mean$coefficients), c(fit_zero$coefficients))
  } else if(family=="Multinomial"){
    tab = cbind(c(betas), c(w), c(rbind(rep(NA,P),coefficients(fit_mean))), c(rbind(rep(NA,P),coefficients(fit_zero))))
  }
  colnames(tab) = c("Truth","dlglm","mean","zero")
  
  p<-tableGrob(round(tab,3))
  title <- textGrob("Wts from dlglm, coefs from mean/zero imp + glm",gp=gpar(fontsize=9))
  padding <- unit(5,"mm")
  
  p <- gtable_add_rows(
    p, 
    heights = grobHeight(title) + padding,
    pos = 0)
  p <- gtable_add_grob(
    p, 
    title, 
    1, 1, 1, ncol(p))
  ggsave(p,file=sprintf("%s/Diagnostics/%s_coefs_tab.png",dir_name,mechanism), width=6, height=6, units="in")
  
  rel_diff = unlist(abs(Ys$test - mu_y)/max(abs(Ys$test),0.001))
  rel_diff_mean = unlist(abs(Ys$test - yhat_mean_pred)/max(abs(Ys$test),0.001))
  rel_diff_zero = unlist(abs(Ys$test - yhat_zero_pred)/max(abs(Ys$test),0.001))
  
  tab2=matrix(ncol=3,nrow=1); colnames(tab2) = c("dlglm","mean","zero")
  tab2[1,1] = mean(rel_diff)
  tab2[1,2] = mean(rel_diff_mean)
  tab2[1,3] = mean(rel_diff_zero)
  
  tiff(filename=sprintf("%s/Diagnostics/%s_predY_rel.tiff",dir_name,mechanism),res = 300,width = 6, height = 6, units = 'in')
  boxplot(rel_diff,rel_diff_mean,rel_diff_zero, outline=F,
          names=c("dlglm","mean","zero"),
          main="Relative Difference between test Y and predicted Y")
  dev.off()

  p<-tableGrob(round(tab2,3))
  title <- textGrob("Rel diff between true and pred Ys",gp=gpar(fontsize=8.5))
  p <- gtable_add_rows(
    p, 
    heights = grobHeight(title) + padding,
    pos = 0)
  p <- gtable_add_grob(
    p, 
    title, 
    1, 1, 1, ncol(p))
  ggsave(p,file=sprintf("%s/Diagnostics/%s_mean_pctdiff_predY.png",dir_name,mechanism), width=6, height=4, units="in")

  ###########################################
  ###### look at imputation in missing ######
  ###########################################
  xhat = res$xhat
  # xfull = res$xfull
  for(i in 1:ncol(xhat)){
    if(normalize){
      xhat[,i] = res$xhat[,i] * norm_sds_x[i] + norm_means_x[i]
    }else{
      xhat[,i] = res$xhat[,i]
    }
    # xfull[,i] = res$xfull[,i] * norm_sds_x[i] + norm_means_x[i]
  }
  if(normalize){ yhat = res$yhat* norm_sd_y + norm_mean_y } else{ yhat = res$yhat }
  if(grepl("x",case)){
    if(all(Rxs$test==1)){stop("case states missing x, but no missing x")}
    L1s_x = abs(Xs$test-xhat)[Rxs$test==0]
    L1_x = mean(abs(Xs$test-xhat)[Rxs$test==0])
    pcts_x = (abs(Xs$test-xhat)/abs(Xs$test))[Rxs$test==0]
    boxplot(L1s_x,outline=F, main="Absolute deviation between true and imputed X")
  }
  if(grepl("y",case)){
    if(all(Rys$test==1)){stop("case states missing y, but no missing y")}
    L1s_y = abs(as.matrix(Ys$test)-yhat)[Rys$test==0]
    L1_y = mean(abs(as.matrix(Ys$test)-yhat)[Rys$test==0])
    pcts_y = (abs(as.matrix(Ys$test)-yhat)/abs(as.matrix(Ys$test)))[Rys$test==0]
    boxplot(L1s_y,outline=F, main="Absolute deviation between true and imputed Y")
  }
  
  # Mean/zero imputation results:
  
  if(grepl("x",case)){
    diffs_x = abs(Xs$test-xhat); diffs_mean_x = abs(Xs$test-xhat_mean); diffs_zero_x = abs(Xs$test-xhat_zero)
    L1s_mean_x = abs(Xs$test-xhat_mean)[Rxs$test==0]; L1_mean_x = mean(abs(Xs$test-xhat_mean)[Rxs$test==0])
    L1s_zero_x = abs(Xs$test-xhat_zero)[Rxs$test==0]; L1_zero_x = mean(abs(Xs$test-xhat_zero)[Rxs$test==0])
    pcts_mean_x = (abs(Xs$test-xhat_mean)/abs(Xs$test))[Rxs$test==0]
    pcts_zero_x = (abs(Xs$test-xhat_zero)/abs(Xs$test))[Rxs$test==0]
    boxplot(L1s_mean_x,outline=F, main="Absolute deviation between true and mean imputed X")
    boxplot(L1s_zero_x,outline=F, main="Absolute deviation between true and zero imputed X")
  }
  if(grepl("y",case)){
    L1s_mean_y = abs(as.matrix(Ys$test)-yhat_mean)[Rys$test==0]; L1_mean_y = mean(abs(as.matrix(Ys$test)-yhat_mean)[Rys$test==0])
    L1s_zero_y = abs(as.matrix(Ys$test)-yhat_zero)[Rys$test==0]; L1_zero_y = mean(abs(as.matrix(Ys$test)-yhat_zero)[Rys$test==0])
    pcts_mean_y = (abs(as.matrix(Ys$test)-yhat_mean)/abs(Ys$test))[Rys$test==0]
    pcts_zero_y = (abs(as.matrix(Ys$test)-yhat_zero)/abs(Ys$test))[Rys$test==0]
    boxplot(L1s_mean_y,outline=F, main="Absolute deviation between true and mean imputed Y")
    boxplot(L1s_zero_y,outline=F, main="Absolute deviation between true and zero imputed Y")
  }
  
  # compare both imputation results
  if(grepl("x",case)){
    for(c in miss_cols){
      ids = Rxs$test[,c]==0
      tiff(filename=sprintf("%s/Diagnostics/%s_imputedX_col%d.tiff",dir_name,mechanism,c),res = 300,width = 5, height = 5, units = 'in')
      boxplot(diffs_x[ids,c], diffs_mean_x[ids,c], diffs_zero_x[ids,c],
              names=c("dlglm","mean","zero"),
              outline=F, main=sprintf("Abs dev between true and imputed X in col%d",c))
      dev.off()
    }
    tiff(filename=sprintf("%s/Diagnostics/%s_imputedX.tiff",dir_name,mechanism),res = 300,width = 5, height = 5, units = 'in')
    boxplot(L1s_x,L1s_mean_x,L1s_zero_x,
            names=c("dlglm","mean","zero"),
            outline=F, main="Absolute deviation between true and imputed X")
    dev.off()
    tiff(filename=sprintf("%s/Diagnostics/%s_imputedpctX.tiff",dir_name,mechanism),res = 300,width = 5, height = 5, units = 'in')
    boxplot(pcts_x,pcts_mean_x,pcts_zero_x,
            names=c("dlglm","mean","zero"),
            outline=F, main="Relative difference between true and imputed X")
    dev.off()
    
    tiff(filename=sprintf("%s/Diagnostics/%s_imputedX_refzero1.tiff",dir_name,mechanism),res = 300,width = 5, height = 5, units = 'in')
    boxplot(L1s_x-L1s_zero_x,
            names=c("dlglm"),
            outline=F, main="Err dlglm-imputed X - Err zero-imputed X")
    abline(h=L1s_mean_x[1]-L1s_zero_x[1], col = "Red")
    dev.off()
    tiff(filename=sprintf("%s/Diagnostics/%s_imputedX_refzero2.tiff",dir_name,mechanism),res = 300,width = 5, height = 5, units = 'in')
    boxplot(L1s_x-L1s_zero_x,L1s_mean_x-L1s_zero_x,
            names=c("dlglm","mean"),
            outline=F, main="Err dlglm/mean-imputed X - Err zero-imputed X")
    abline(h=L1s_mean_x[1]-L1s_zero_x[1], col = "Red")
    dev.off()
    
    
    tiff(filename=sprintf("%s/Diagnostics/%s_scatter_imputedX_refzero.tiff",dir_name,mechanism),res = 300,width = 5, height = 5, units = 'in')
    plot(x=L1s_x-L1s_zero_x,y=L1s_mean_x-L1s_zero_x, main="Err dlglm vs mean imputed X, wrt zero-imputed X")
    dev.off()
    
    tab3=matrix(ncol=3,nrow=1); colnames(tab3) = c("dlglm","mean","zero")
    tab3[1,1] = mean(L1s_x)
    tab3[1,2] = mean(L1s_mean_x)
    tab3[1,3] = mean(L1s_zero_x)
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
    ggsave(p,file=sprintf("%s/Diagnostics/%s_mean_L1s_X.png",dir_name,mechanism), width=6, height=4, units="in")
  }
  if(grepl("y",case)){
    tiff(filename=sprintf("%s/Diagnostics/%s_imputedY.tiff",dir_name,mechanism),res = 300,width = 5, height = 5, units = 'in')
    boxplot(L1s_y,L1s_mean_y,L1s_zero_y,
            names=c("dlglm","mean","zero"),
            outline=F, main="Absolute deviation between true and imputed Y")
    dev.off()
    tiff(filename=sprintf("%s/Diagnostics/%s_imputedpctY.tiff",dir_name,mechanism),res = 300,width = 5, height = 5, units = 'in')
    boxplot(pcts_y,pcts_mean_y,pcts_zero_y,
            names=c("dlglm","mean","zero"),
            outline=F, main="Relative difference between true and imputed Y")
    dev.off()
    
    tiff(filename=sprintf("%s/Diagnostics/%s_imputedY_refzero.tiff",dir_name,mechanism),res = 300,width = 5, height = 5, units = 'in')
    boxplot(L1s_y-L1s_zero_y,
            names=c("dlglm"),
            outline=F, main="Err imputed Y - Err zero-imputed Y")
    abline(h=L1s_mean_y[1]-L1s_zero_y[1], col = "Red")
    dev.off()
  }
  
  ## TRYING: glm on dlglm resulting xhat (test data)
  # d_dlglm = cbind( xhat, Ys$test )
  # fit_dlglm = glm(Y ~ 0 + . , data=d_dlglm)
  
  # mean - dlglm pairs plot
  library(GGally)
  my_fn = function(data, mapping, ...){
    p <- ggplot(data=data,mapping=mapping) + geom_point() + geom_abline(slope=1, intercept=0, colour="red")
    p
  }
  # for(c in miss_cols){
    # mask_ids = Rxs$test[,c]==0
    p = ggpairs(data.frame(Ys$test,
                             "dlglm"=mu_y,
                             "mean"=yhat_mean_pred,
                             "zero"=yhat_zero_pred),lower=list(continuous=my_fn),
                title="Plots of true vs predicted values of Y")
    ggsave(sprintf("%s/Diagnostics/%s_pairs.tiff",dir_name,mechanism),p,height=6, width=6, units="in")
    df = data.frame("dlglm"=Ys$test-mu_y,
                    "mean"=Ys$test-yhat_mean_pred,
                    "zero"=Ys$test-yhat_zero_pred)
    names(df) = c("dlglm","mean","zero")
    p = ggpairs(df,lower=list(continuous=my_fn),
                title="Plots of differences between true vs each set of predicted values of Y")
    ggsave(sprintf("%s/Diagnostics/%s_pairs_diff.tiff",dir_name,mechanism),p,height=6, width=6, units="in")
    df=data.frame("dlglm"=abs(Ys$test-mu_y),
                  "mean"=abs(Ys$test-yhat_mean_pred),
                  "zero"=abs(Ys$test-yhat_zero_pred))
    names(df) = c("dlglm","mean","zero")
    p = ggpairs(df,lower=list(continuous=my_fn),
                title="Plots of absolute differences between true vs each set of predicted values of Y")
    ggsave(sprintf("%s/Diagnostics/%s_pairs_absdiff.tiff",dir_name,mechanism),p,height=6, width=6, units="in")
  # }
  
}

# prefix="Xmean4/"
# prefix="Xmean10sd2_beta5_pi50/"
# mu=0; sd=1; beta=5; pi=0.5
# mu=1; sd=2; beta=5; pi=0.5
mu=5; sd=5; beta=5; pi=0.5
prefix=sprintf("Xmean%dsd%d_beta%d_pi%d/",mu,sd,beta,pi*100)
# prefix=sprintf("Xmean%dsd%d_beta%d_pi%d_corr0.2/",mu,sd,beta,pi*100)
# N=1e5; P=8; phi0=100
N=1e5; P=8; phi0=5
data.file.name = NULL; mask.file.name=NULL
case="x"; normalize=F
data_type_x="real"; data_type_y = "real"   # real, cat, cts
# data_type_x="real"; data_type_y = "cat"; C=3   # real, cat, cts

sim_indexes = 1; mechanisms=c("MNAR")

for(s in 1:length(sim_indexes)){for(m in 1:length(mechanisms)){
  
  sim_index=sim_indexes[s]; mechanism=mechanisms[m]
  sim.params = list(N=N, P=P, data_types=rep(data_type_x,P),
                    family=if(data_type_y=="real"){"Gaussian"}else if(data_type_y=="cat"){"Multinomial"}else if(data_typ_y=="cts"){"Poisson"},
                    sim_index=sim_index, ratios=c(train=.6,valid=.2,test=.2),
                    mu=mu, sd=sd, beta=beta, C=C)
  miss.params = list(scheme="UV", mechanism=mechanism, pi=pi, phi0=phi0)
  
  processResults(prefix=prefix,data.file.name = data.file.name, mask.file.name=mask.file.name,
                 sim.params = sim.params,
                 miss.params = miss.params,
                 case=case, normalize=normalize,
                 data_type_x=data_type_x, data_type_y = data_type_y)
  
}}
