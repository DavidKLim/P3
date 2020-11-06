def dlglm(X,Rx,Y,Ry, covars_r_x, covars_r_y, norm_means_x, norm_sds_x, norm_mean_y, norm_sd_y, learn_r, Ignorable=False, family="Gaussian", link="identity", impute_bs=None,arch="IWAE",add_miss_term=False,draw_miss=True,pre_impute_value=0,n_hidden_layers=2,n_hidden_layers_r=0,h1=8,h2=8,h3=0,phi0=None,phi=None,train=1,saved_model=None,sigma="elu",bs = 64,n_epochs = 2002,lr=0.001,L=20,M=20,trace=False):
  #family="Gaussian"; link="identity"
  #family="Multinomial"; link="mlogit"
  #family="Poisson"; link="log"
  # covars_r_x: vector of P: 1/0 for inclusion/exclusion of each feature as covariate of missingness model
  # covars_r_y: 1 or 0

  temp = 0.5; temp_min=0.5

  if (h2 is None) and (h3 is None):
    h2=h1; h3=h1
  import torch     # this module not found in Longleaf
  # import torchvision
  import torch.nn as nn
  import numpy as np
  import scipy.stats
  import scipy.io
  import scipy.sparse
  import pandas as pd
  import matplotlib.pyplot as plt
  import torch.distributions as td
  from torch import nn, optim
  from torch.nn import functional as F
  #import torch.nn.utils.prune as prune
  # from torchvision import datasets, transforms
  # from torchvision.utils import save_image
  import time

  from torch.distributions import constraints
  from torch.distributions.distribution import Distribution
  from torch.distributions.utils import broadcast_all
  import torch.nn.functional as F
  from torch.autograd import Variable
  import torch.nn.utils.prune as prune
  from collections import OrderedDict
  import os
  os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

  if np.all(Rx==1) and np.all(Ry==1):
    Ignorable=True
  
  data_types_x = np.repeat("real", X.shape[1])
  #family="Gaussian"   # p(y|x) family
  #link="identity"     # g(E[y|x]) = eta
  
  if (family=="Multinomial"):
    C = len(np.unique(Y[~np.isnan(Y)]))   # if we're looking at categorical data, then determine #categories by unique values (nonmissing) in Y
    print("# classes: " + str(C))

  # Figure out the correct case:
  miss_x = False; miss_y = False
  if np.sum(Rx==0)>0: miss_x = True
  if np.sum(Ry==0)>0: miss_y = True

  #if (not (covars_miss==None).all()):
  #  covars=True
  #  pr1 = np.shape(covars_miss)[1]
  #else:
  #  covars=False
  #  pr1=0
  covars_miss = None ; covars=False  # turned off for now

  # input_r: "r" or "pr" --> what to input into NNs for mask, r (1/0) or p(r=1) (probs)
  # do "r" only for now
  def mse(xhat,xtrue,mask):
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return {'miss':np.mean(np.power(xhat-xtrue,2)[mask<0.5]),'obs':np.mean(np.power(xhat-xtrue,2)[mask>0.5])}
    #return {'miss':np.mean(np.power(xhat-xtrue,2)[~mask]),'obs':np.mean(np.power(xhat-xtrue,2)[mask])}
  
  #xfull = (data - np.mean(data,0))/np.std(data,0)
  xfull = (X - norm_means_x)/norm_sds_x    # need to not do this if data type is not Gaussian
  if family=="Gaussian":
    yfull = (Y - norm_mean_y)/norm_sd_y
  else: yfull = Y.astype("float")
  
  # Loading and processing data
  n = xfull.shape[0] # number of observations
  p = xfull.shape[1] # number of features
  np.random.seed(1234)

  if (impute_bs==None): impute_bs = n       # if number of observations to feed into imputation, then impute all simultaneously (may be memory-inefficient)
  
  xmiss = np.copy(xfull)
  xmiss[Rx==0]=np.nan
  mask_x = np.isfinite(xmiss) # binary mask that indicates which values are missing

  ymiss = np.copy(yfull)
  ymiss[Ry==0]=np.nan
  mask_y = np.isfinite(ymiss)

  yhat_0 = np.copy(ymiss) ### later change this to ymiss with missing y pre-imputed
  xhat_0 = np.copy(xmiss)

  # Custom pre-impute values
  if (pre_impute_value == "mean_obs"):
    xhat_0[Rx==0] = np.mean(xmiss[Rx==1],0); yhat_0[Ry==0] = np.mean(ymiss[Ry==1],0)
  elif (pre_impute_value == "mean_miss"):
    xhat_0[Rx==0] = np.mean(xmiss[Rx==0],0); yhat_0[Ry==0] = np.mean(ymiss[Ry==0],0)
  elif (pre_impute_value == "truth"):
    xhat_0 = np.copy(xfull); yhat_0 = np.copy(yfull)
  else:
    xhat_0[np.isnan(xmiss)] = pre_impute_value; yhat_0[np.isnan(ymiss)] = pre_impute_value

  init_mse = mse(xfull,xhat_0,mask_x)
  print("Pre-imputation MSE (obs, should be 0): " + str(init_mse['obs']))
  print("Pre-imputation MSE (miss): " + str(init_mse['miss']))

  prx = np.sum(covars_r_x).astype(int)
  pry = np.sum(covars_r_y).astype(int)
  pr = prx + pry
  if not learn_r: phi=torch.from_numpy(phi).float().cuda()
  
  # Define decoder/encoder
  if (sigma=="relu"): act_fun=torch.nn.ReLU()
  elif (sigma=="elu"): act_fun=torch.nn.ELU()

  full_obs_ids = np.sum(Rx==0,axis=0)==0    # columns that are fully observed need not have missingness modelled
  miss_ids = np.sum(Rx==0,axis=0)>0

  p_miss = np.sum(~full_obs_ids)
  
  n_params_xm = 2*p # Gaussian (mean, sd. p features in X)
  if family=="Gaussian":
    n_params_ym = 2
    #n_params_y = 2 # Gaussian (mean, sd. One feature in y)
    n_params_y = 1 # Gaussian (just the mean. learn the SDs (subj specific) directly as parameters like mu_x and sd_x)
  elif family=="Multinomial":
    n_params_ym = C    # probs for each of the K classes
    n_params_y = C
  elif family=="Poisson":
    n_params_ym = 1
    n_params_y = 1
  n_params_r = p_miss*(miss_x) + 1*(miss_y) # Bernoulli (prob. p features in X) --> 1 if missing in y and not X


  def network_maker(act_fun, n_hidden_layers, in_h, h, out_h, bias=True, dropout=False):
    # create NN layers
    if n_hidden_layers==0:
      layers = [ nn.Linear(in_h, out_h, bias), ]
    elif n_hidden_layers>0:
      layers = [ nn.Linear(in_h , h, bias), act_fun, ]
      for i in range(n_hidden_layers-1):
        layers.append( nn.Linear(h, h, bias), )
        layers.append( act_fun, )
      layers.append(nn.Linear(h, out_h, bias))
    elif n_hidden_layers<0:
      raise Exception("n_hidden_layers must be >= 0")
    
    # insert dropout layer (if applicable)
    if dropout:
      layers.insert(0, nn.Dropout())
    
    # create NN
    model = nn.Sequential(*layers)
    
    # initialize weights
    def weights_init(layer):
      if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
    model.apply(weights_init)
    
    return model
  
  if miss_x: NN_xm = network_maker(act_fun, n_hidden_layers, 2*p, h1, n_params_xm, False, True).cuda()
  else: NN_xm = None
  if miss_y: NN_ym = network_maker(act_fun, n_hidden_layers, p+2, h1, n_params_ym, False, True).cuda()
  else: NN_ym = None
  NN_y = network_maker(act_fun, 0, p, h2, n_params_y, False, False).cuda()

  if not Ignorable: NN_r = network_maker(act_fun, n_hidden_layers_r, pr, h3, n_params_r, False, True).cuda()
  else: NN_r=None
  # can initialize NN_ym if missingness detected in y , NN_xm if missingness detected in x

  # Prior p(x): mean and sd for each feature
  mu_x = torch.zeros(p, requires_grad=True, device="cuda:0"); scale_x = torch.ones(p, requires_grad=True, device="cuda:0")
  alpha = torch.ones(1, requires_grad=True, device="cuda:0")   # learned directly

  def invlink(link="identity"):
    if link=="identity":
      fx = torch.nn.Identity(0)
    elif link=="log":
      fx = torch.exp
    elif link=="logit":
      fx = torch.nn.Sigmoid()
    elif link=="mlogit":
      fx = torch.nn.Softmax(dim=1)   # same as sigmoid, except imposes that the probs sum to 1 across classes
    return fx
  
  def V(mu, alpha, family="Gaussian"):
    #print(mu.shape)
    if family=="Gaussian":
      out = alpha*torch.ones([mu.shape[0]]).cuda()
    elif family=="Poisson":
      out = mu
    elif family=="NB":
      out = mu + alpha*torch.pow(mu, 2).cuda()
    elif family=="Binomial":
      out = mu*(1-(mu/n_successes))
    elif family=="Multinomial":
      out = mu*(1-mu)
    return out

  #p_x = td.Normal(loc=mu_x, scale=torch.nn.Softplus()(scale_x)+0.001)

  def forward(iota_xfull, iota_x, iota_y, mask_x, mask_y, batch_size, niw):
    tiled_iota_x = torch.Tensor.repeat(iota_x,[niw,1]); tiled_tiled_iota_x = torch.Tensor.repeat(tiled_iota_x,[niw,1])
    tiledmask_x = torch.Tensor.repeat(mask_x,[niw,1]); tiled_tiledmask_x = torch.Tensor.repeat(tiledmask_x,[niw,1])
    if not draw_miss: tiled_iota_xfull = torch.Tensor.repeat(iota_xfull,[niw,1])
    tiled_iota_y = torch.Tensor.repeat(iota_y,[niw,1]); tiled_tiled_iota_y = torch.Tensor.repeat(tiled_iota_y,[niw,1])
    tiledmask_y = torch.Tensor.repeat(mask_y,[niw,1]); tiled_tiledmask_y = torch.Tensor.repeat(tiledmask_y,[niw,1])
    if not draw_miss: tiled_iota_yfull = torch.Tensor.repeat(iota_yfull,[niw,1])
    
    p_x = td.Normal(loc=mu_x, scale=torch.nn.Softplus()(scale_x)+0.001)

    params_x = None; xm = iota_x; xm_flat = torch.Tensor.repeat(iota_x,[niw,1])  # if no missing x
    params_y = None; ym = iota_y; ym_flat = torch.Tensor.repeat(iota_y,[niw,1])

    ## NN_xm ## p(xm|xo,r)    (if missing in x detected)
    if miss_x:
      out_NN_xm = NN_xm(torch.cat([iota_x,mask_x],1))
      # bs x p -- > sample niw times
      qxmgivenxor = td.Normal(loc=out_NN_xm[..., :p],scale=torch.nn.Softplus()(out_NN_xm[..., p:(2*p)])+0.001)    ### condition contribution of this term in the ELBO by miss_x
      params_xm = {'mean':out_NN_xm[..., :p], 'scale':torch.nn.Softplus()(out_NN_xm[..., p:(2*p)])+0.001}
      if draw_miss: xm = qxmgivenxor.rsample([niw]); xm_flat = xm.reshape([niw*batch_size,p])
    else: 
      qxmgivenxor=None; params_xm=None; xm_flat = torch.Tensor.repeat(iota_x,[niw,1])
    # organize completed (sampled) xincluded for missingness model. observed values are not sampled
    if miss_x:
      if miss_y:
        tiled_xm_flat = torch.Tensor.repeat(xm_flat,[niw,1])
        xincluded = tiled_tiled_iota_x*(tiled_tiledmask_x) + tiled_xm_flat*(1-tiled_tiledmask_x)
      else:
        xincluded = tiled_iota_x*(tiledmask_x) + xm_flat*(1-tiledmask_x)
    else:
      xincluded = iota_x

    ## NN_ym ## p(ym|yo,x,r)   (if missing in y detected)    
    if miss_y:
      if not miss_x:
        out_NN_ym = NN_ym(torch.cat([iota_y, iota_x, mask_y],1))
        # bs x 1 --> sample niw times
      elif miss_x:
        out_NN_ym = NN_ym(torch.cat([tiled_iota_y, tiledmask_x*tiled_iota_x + (1-tiledmask_x)*xm_flat, tiledmask_y],1))
        # (niw*bs) x 1 --> sampled niw times
      if family=="Gaussian":
        qymgivenyor = td.Normal(loc=out_NN_ym[..., :1],scale=torch.nn.Softplus()(out_NN_ym[..., 1:2])+0.001)     ### condition contribution of this term in the ELBO by miss_y
        params_ym = {'mean':out_NN_ym[..., :1], 'scale':torch.nn.Softplus()(out_NN_ym[..., 1:2])+0.001}
      if draw_miss: ym = qymgivenyor.rsample([niw]); ym_flat = ym.reshape([-1,1])    # ym_flat is (niw*bs x 1) if no miss_x, and (niw*niw*bs x 1) if miss_x
    else:
      qymgivenyor=None; params_ym=None; ym_flat = torch.Tensor.repeat(iota_y,[niw,1])
    
    # organize completed (sampled) xincluded for missingness model. observed values are not sampled
    if miss_y:
      if miss_x:  yincluded = tiled_tiled_iota_y*(tiled_tiledmask_y) + ym_flat*(1-tiled_tiledmask_y)
      else:  yincluded = tiled_iota_y*(tiledmask_y) + ym_flat*(1-tiledmask_y)
    else:
      if miss_x:  yincluded = tiled_iota_y
      else:  yincluded = iota_y

    ## NN_y ##      p(y|x)
    out_NN_y = NN_y(xincluded)     # if miss_x and miss_y: this becomes niw*niw*bs x p, otherwise: niw*bs x p
    if family=="Gaussian":
      mu_y = invlink(link)(out_NN_y[..., 0]);  var_y = V(mu_y, torch.nn.Softplus()(alpha)+0.001, family)   # default: link="identity", family="Gaussian"
      pygivenx = td.Normal(loc = mu_y, scale = (var_y)**(1/2))    # scale = sd = var^(1/2)
      params_y = {'mean': mu_y.detach(), 'scale': (var_y.detach())**(1/2)}
    elif family=="Multinomial":
      probs = invlink(link)(out_NN_y[..., :C])
      pygivenx = td.OneHotCategorical(probs=probs)
      #print("probs:"); print(probs)
      #print("pygivenx (event_shape):"); print(pygivenx.event_shape)
      #print("pygivenx (batch_shape):"); print(pygivenx.batch_shape)
      params_y = {'probs': probs.detach()}
    elif family=="Poisson":
      lambda_y = invlink(link)(out_NN_y[..., 0])  # variance is the same as mean in Poisson
      pygivenx = td.Poisson(rate = lambda_y)
      params_y = {'lambda': lambda_y.detach()}

    #print(pygivenx.rsample().shape)

    ## NN_r ##   p(r|x,y,covars): always. Include option to specify covariates in X, y, and additional covars_miss
    # Organize covariates for missingness model (NN_r)
    if covars_r_y==1:
      if np.sum(covars_r_x)>0: covars_included = torch.cat([xincluded[:,covars_r_x==1], yincluded],1)
      else: covars_included = yincluded
    elif covars_r_y==0:
      if np.sum(covars_r_x)>0: covars_included = xincluded[:,covars_r_x==1]
      # else: IGNORABLE HERE. NO COVARIATES
    
    #print(covars_included.shape)
    #print(NN_r)
    if not Ignorable:
      if (covars): out_NN_r = NN_r(torch.cat([covars_included, covars_miss]))   # right now: just X in as covariates (Case 1)    (niw*niw*bs x p) for case 3, (niw*bs x p) for other cases
      else: out_NN_r = NN_r(covars_included)        # can additionally include covariates
      prgivenxy = td.Bernoulli(logits = out_NN_r)   # for just the features with missing valuess
      params_r = {'probs': torch.nn.Sigmoid()(out_NN_r).detach()}
    else: prgivenxy=None; params_r=None


    return xincluded, yincluded, p_x, qxmgivenxor, qymgivenyor, pygivenx, prgivenxy, params_xm, params_ym, params_y, params_r
  # return xincluded, yincluded, p_x, qxmgivenxor, qymgivenyor, pygivenx, prgivenxy, params_xm, params_ym, params_y, params_r
  
  # cases (1): miss_x, (2) miss_y, (3) miss_x and miss_y
  # xincluded: (1) niw*bs x p, (2) niw*bs x p, (3) niw*niw*bs x p
  # yincluded: (1) niw*bs x p, (2) niw*bs x p, (3) niw*niw*bs x p
  # p_x: 0 x p
  # qxmgivenxor: (1) bs x p, (2) None, (3) bs x p
  # qymgivenyor: (1) None, (2) bs x p, (3) (niw*bs) x p
  # pygivenx: (1) niw*bs x p, (2) niw*bs x p, (3) niw*niw*bs x p
  # prgivenxy: (1) niw*bs x p, (2) niw*bs x p, (3) niw*niw*bs x p
   
  def compute_loss(iota_xfull, iota_yfull, iota_x, iota_y, mask_x, mask_y, covar_miss, temp):
    batch_size = iota_x.shape[0]
    tiled_iota_x = torch.Tensor.repeat(iota_x,[M,1]).cuda(); tiled_iota_y = torch.Tensor.repeat(iota_y,[M,1]).cuda()
    tiled_tiled_iota_x = torch.Tensor.repeat(tiled_iota_x,[M,1]).cuda(); tiled_tiled_iota_y = torch.Tensor.repeat(tiled_iota_y,[M,1]).cuda()
    tiledmask_x = torch.Tensor.repeat(mask_x,[M,1]).cuda(); tiled_tiledmask_x = torch.Tensor.repeat(tiledmask_x,[M,1]).cuda()
    tiledmask_y = torch.Tensor.repeat(mask_y,[M,1]).cuda(); tiled_tiledmask_y = torch.Tensor.repeat(tiledmask_y,[M,1]).cuda()
    if add_miss_term or not draw_miss: tiled_iota_xfull = torch.Tensor.repeat(iota_xfull,[M,1]).cuda(); tiled_iota_yfull = torch.Tensor.repeat(iota_yfull,[M,1]).cuda()
    else: tiled_iota_xfull = None
    
    if covars: tiled_covars_miss = torch.Tensor.repeat(covar_miss,[M,1])
    else: tiled_covars_miss=None

    xincluded, yincluded, p_x, qxmgivenxor, qymgivenyor, pygivenx, prgivenxy, params_xm, params_ym, params_y, params_r = forward(iota_xfull, iota_x, iota_y, mask_x, mask_y, batch_size, M)
    
    if family=="Multinomial":
      #!print(yincluded[:20]); print(pygivenx); print(M); print(batch_size)
      #yincluded2 = np.zeros([yincluded.shape[0], int(torch.max(yincluded).item())+1])
      #yincluded2[np.arange(yincluded.shape[0]), yincluded.cpu().data.numpy().astype("int")] = 1
      yincluded=torch.nn.functional.one_hot(yincluded.to(torch.int64)).reshape([-1,C])
      #print(yincluded2)
    
    # form of ELBO: log p(y|x) + log p(x) + log p(r|x) - log q(xm|xo, r)

    ## COMPUTE LOG PROBABILITIES ##
    if miss_x and miss_y:     # case 3
      # log p(r|x,y) # niw*niw*bs x p
      if not Ignorable:
        all_logprgivenxy = prgivenxy.log_prob(torch.cat([tiled_tiledmask_x[:,miss_ids], tiled_tiledmask_y],1))
        logprgivenxy = torch.sum(all_logprgivenxy,1).reshape([M*M,batch_size])
      else: all_logprgivenxy=0; logprgivenxy=0
      # log p(y|x)   # niw*niw*bs x p
      if family=="Gaussian": all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1]))    #yincluded: M*M*batch_size
      else: all_log_pygivenx = pygivenx.log_prob(yincluded)

      logpygivenx = all_log_pygivenx.reshape([M*M,batch_size])
      # log p(x)
      logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([M*M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
      ##logpx = torch.sum(p_x.log_prob(xincluded)*(1-tiled_tiledmask_x),axis=1).reshape([M*M,batch_size])     # xincluded: xo and sample of xm (just missing x)
      # log q(xm|xo,r)
      logqxmgivenxor = torch.sum(qxmgivenxor.log_prob(xincluded.reshape([M*M,batch_size,p])).reshape([M*M*batch_size,p])*(1-tiled_tiledmask_x),1).reshape([M*M,batch_size])
      # log q(ym|yo,r,xm,xo)
      logqymgivenyor = (qymgivenyor.log_prob(yincluded.reshape([M,-1,1])).reshape([M*M*batch_size,1])*(1-tiled_tiledmask_y)).reshape([M*M,batch_size])
    else:
      # log p(r|x,y)
      if miss_x and not miss_y:  # case 1
        if not Ignorable:
          all_logprgivenxy = prgivenxy.log_prob(tiledmask_x[:,miss_ids])  # M*bs x p
          logprgivenxy = torch.sum(all_logprgivenxy,1).reshape([M,batch_size])
        else: all_logprgivenxy=0; logprgivenxy=0
        # log q(xm|xo,r)
        logqxmgivenxor = torch.sum(qxmgivenxor.log_prob(xincluded.reshape([M,batch_size,-1])).reshape([M*batch_size,-1])*(1-tiledmask_x),1).reshape([M,batch_size]); logqymgivenyor=0
        # log p(y|x)   # all cases set to be (niw*niw*bs) x p

        
        if family=="Multinomial": all_log_pygivenx = pygivenx.log_prob(yincluded)
        else: all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1])) #yincluded: M*batch_size

        #print("Diagnostics:"); print(yincluded.shape); print(pygivenx.event_shape); print(pygivenx.batch_shape); print(all_log_pygivenx.shape)
        logpygivenx = all_log_pygivenx.reshape([M,batch_size])
        # log p(x)
        logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
      elif not miss_x and miss_y: # case 2
        if not Ignorable:
          all_logprgivenxy = prgivenxy.log_prob(tiledmask_y)
          logprgivenxy = all_logprgivenxy.reshape([M,batch_size])   # no need to sum across columns (missingness of just y)
        else: all_logprgivenxy=0; logprgivenxy=0
        # log q(ym|xo,r)
        logqymgivenyor = (qymgivenyor.log_prob(yincluded.reshape([M,-1,1])).reshape([M*batch_size,1])*(1-tiledmask_y)).reshape([M,batch_size]); logqxmgivenxor=0
        # log p(y|x)   # all cases set to be (niw*niw*bs) x p
        
        if family=="Multinomial": all_log_pygivenx = pygivenx.log_prob(yincluded)
        else: all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1]))

        logpygivenx = all_log_pygivenx.reshape([M,batch_size])
        # log p(x)
        logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
      else:     # no missing
        logprgivenxy=0; all_logprgivenxy=0; logqxmgivenxor=0; logqymgivenyor=0

        if family=="Multinomial": all_log_pygivenx = pygivenx.log_prob(yincluded)
        else: all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1]))

        logpygivenx = all_log_pygivenx.reshape([1,batch_size])
        # log p(x)
        logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([1,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
      
      
      ##logpx = torch.sum(p_x.log_prob(xincluded)*(1-tiledmask_x),axis=1).reshape([M,batch_size])     # xincluded: xo and sample of xm (just missing x)
    
    if not Ignorable: sum_logpr = np.sum(logprgivenxy.cpu().data.numpy())
    else: sum_logpr = 0

    sum_logpygivenx = np.sum(logpygivenx.cpu().data.numpy())
    sum_logpx = np.sum(logpx.cpu().data.numpy())
    if miss_x: sum_logqxm = np.sum(logqxmgivenxor.cpu().data.numpy())
    else: sum_logqxm = 0; logqxmgivenxor=0
    if miss_y: sum_logqym = np.sum(logqymgivenyor.cpu().data.numpy())
    else: sum_logqym = 0; logqymgivenyor=0

    #c = torch.cuda.memory_cached(0); a = torch.cuda.memory_allocated(0)
    #print("memory free:"); print(c-a)  # free inside cache


    # log q(ym|yo,r) ## add this into ELBO too
    # Case 1: x miss, y obs --> K samples of X
    # Case 2: x obs, y miss --> K samples of Y
    # Case 3: x miss, y miss --> K samples of X, K*M samples of Y. NEED TO MAKE THIS CONSISTENT: just make K=M and M samples of X and M samples of Y

    if arch=="VAE":
      ## VAE NEGATIVE LOG-LIKE ##
      neg_bound = -torch.mean(logpygivenx + logpx + logprgivenxy - logqxmgivenxor - logqymgivenyor)
    elif arch=="IWAE":
      ## IWAE NEGATIVE LOG-LIKE ##
      neg_bound = np.log(M) + np.log(M)*(miss_x and miss_y) - torch.mean(torch.logsumexp(logpygivenx + logpx + logprgivenxy - logqxmgivenxor - logqymgivenyor,0))
    
    return{'neg_bound':neg_bound, 'params_xm': params_xm, 'params_ym': params_ym, 'params_y': params_y, 'params_r':params_r, 'sum_logpr': sum_logpr,'sum_logpygivenx':sum_logpygivenx,'sum_logpx': sum_logpx,'sum_logqxm': sum_logqxm,'sum_logqym': sum_logqym}
  #return{'neg_bound':neg_bound, 'params_xm': params_xm, 'params_ym': params_ym, 'params_y': params_y, 'params_r':params_r, 'sum_logpr': sum_logpr,'sum_logpygivenx': sum_logpygivenx,'sum_logpx': sum_logpx,'sum_logqxm': sum_logqxm,'sum_logqym': sum_logqym}

  def impute(iota_xfull, iota_yfull, iota_x, iota_y, mask_x, mask_y, covar_miss, L, temp):
    batch_size = iota_x.shape[0]
    tiled_iota_x = torch.Tensor.repeat(iota_x,[M,1]).cuda(); tiled_iota_y = torch.Tensor.repeat(iota_y,[M,1]).cuda()
    tiled_tiled_iota_x = torch.Tensor.repeat(tiled_iota_x,[M,1]).cuda(); tiled_tiled_iota_y = torch.Tensor.repeat(tiled_iota_y,[M,1]).cuda()
    tiledmask_x = torch.Tensor.repeat(mask_x,[M,1]).cuda(); tiled_tiledmask_x = torch.Tensor.repeat(tiledmask_x,[M,1]).cuda()
    tiledmask_y = torch.Tensor.repeat(mask_y,[M,1]).cuda(); tiled_tiledmask_y = torch.Tensor.repeat(tiledmask_y,[M,1]).cuda()
    if add_miss_term or not draw_miss: tiled_iota_xfull = torch.Tensor.repeat(iota_xfull,[M,1]).cuda(); tiled_iota_yfull = torch.Tensor.repeat(iota_yfull,[M,1]).cuda()
    else: tiled_iota_xfull = None
    
    if covars: tiled_covars_miss = torch.Tensor.repeat(covar_miss,[M,1])
    else: tiled_covars_miss=None

    xincluded, yincluded, p_x, qxmgivenxor, qymgivenyor, pygivenx, prgivenxy, params_xm, params_ym, params_y, params_r = forward(iota_xfull, iota_x, iota_y, mask_x, mask_y, batch_size, M)
    
    # form of ELBO: log p(y|x) + log p(x) + log p(r|x) - log q(xm|xo, r)

    ## COMPUTE LOG PROBABILITIES ##
    if miss_x and miss_y:     # case 3
      # log p(r|x,y) # niw*niw*bs x p
      if not Ignorable:
        all_logprgivenxy = prgivenxy.log_prob(torch.cat([tiled_tiledmask_x[:,miss_ids], tiled_tiledmask_y],1))
        logprgivenxy = torch.sum(all_logprgivenxy,1).reshape([M*M,batch_size])
      else: all_logprgivenxy=0; logprgivenxy=0
      # log p(y|x)   # niw*niw*bs x p
      if family=="Gaussian": all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1]))    #yincluded: M*M*batch_size
      else: all_log_pygivenx = pygivenx.log_prob(yincluded)

      logpygivenx = all_log_pygivenx.reshape([M*M,batch_size])
      # log p(x)
      logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([M*M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
      ##logpx = torch.sum(p_x.log_prob(xincluded)*(1-tiled_tiledmask_x),axis=1).reshape([M*M,batch_size])     # xincluded: xo and sample of xm (just missing x)
      # log q(xm|xo,r)
      logqxmgivenxor = torch.sum(qxmgivenxor.log_prob(xincluded.reshape([M*M,batch_size,p])).reshape([M*M*batch_size,p])*(1-tiled_tiledmask_x),1).reshape([M*M,batch_size])
      # log q(ym|yo,r,xm,xo)
      logqymgivenyor = (qymgivenyor.log_prob(yincluded.reshape([M,-1,1])).reshape([M*M*batch_size,1])*(1-tiled_tiledmask_y)).reshape([M*M,batch_size])
    else:
      # log p(r|x,y)
      if miss_x and not miss_y:  # case 1
        if not Ignorable:
          all_logprgivenxy = prgivenxy.log_prob(tiledmask_x[:,miss_ids])  # M*bs x p
          logprgivenxy = torch.sum(all_logprgivenxy,1).reshape([M,batch_size])
        else: all_logprgivenxy=0; logprgivenxy=0
        # log q(xm|xo,r)
        logqxmgivenxor = torch.sum(qxmgivenxor.log_prob(xincluded.reshape([M,batch_size,-1])).reshape([M*batch_size,-1])*(1-tiledmask_x),1).reshape([M,batch_size]); logqymgivenyor=0
        # log p(y|x)   # all cases set to be (niw*niw*bs) x p

        
        if family=="Multinomial": all_log_pygivenx = pygivenx.log_prob(yincluded)
        else: all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1])) #yincluded: M*batch_size

        #print("Diagnostics:"); print(yincluded.shape); print(pygivenx.event_shape); print(pygivenx.batch_shape); print(all_log_pygivenx.shape)
        logpygivenx = all_log_pygivenx.reshape([M,batch_size]) #**** THIS IT THE PROBLEM**** ::: #RuntimeError: shape '[5, 5000]' is invalid for input of size 625000000; bs=5000, M=5. IDK what 625000000 came from
        # log p(x)
        logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
      elif not miss_x and miss_y: # case 2
        if not Ignorable:
          all_logprgivenxy = prgivenxy.log_prob(tiledmask_y)
          logprgivenxy = all_logprgivenxy.reshape([M,batch_size])   # no need to sum across columns (missingness of just y)
        else: all_logprgivenxy=0; logprgivenxy=0
        # log q(ym|xo,r)
        logqymgivenyor = (qymgivenyor.log_prob(yincluded.reshape([M,-1,1])).reshape([M*batch_size,1])*(1-tiledmask_y)).reshape([M,batch_size]); logqxmgivenxor=0
        # log p(y|x)   # all cases set to be (niw*niw*bs) x p
        
        if family=="Multinomial": all_log_pygivenx = pygivenx.log_prob(yincluded)
        else: all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1]))

        logpygivenx = all_log_pygivenx.reshape([M,batch_size])
        # log p(x)
        logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([M,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
      else:     # no missing
        logprgivenxy=0; all_logprgivenxy=0; logqxmgivenxor=0; logqymgivenyor=0

        if family=="Multinomial": all_log_pygivenx = pygivenx.log_prob(yincluded)
        else: all_log_pygivenx = pygivenx.log_prob(yincluded.reshape([-1]))

        logpygivenx = all_log_pygivenx.reshape([1,batch_size])
        # log p(x)
        logpx = torch.sum(p_x.log_prob(xincluded),axis=1).reshape([1,batch_size])     # xincluded: xo and sample of xm (both observed and missing x)
      
    if miss_x:
      xmgivenxor = td.Independent(td.Normal(loc=params_xm['mean'],scale=params_xm['scale']),1)
      xms = xmgivenxor.sample([L])
      xm = torch.mean(xms.reshape([L,-1]),0).reshape([batch_size,p]).detach()       # average over L samples of xm. this doesn't weight the samples by how accurate they are
    else:
      xm=None
    if miss_y:
      ymgivenyor = td.Independent(td.Normal(loc=params_ym['mean'],scale=params_ym['scale']),1)
      yms = ymgivenyor.sample([L])
      ym = torch.mean(yms.reshape([L,-1]),0).reshape([-1,1]).detach()       # average over L samples of xm. this doesn't weight the samples by how accurate they are
    else:
      ym=None
    ## SELF-NORMALIZING IMPORTANCE WEIGHTS, USING SAMPLES OF Xm (not sure if this applies here) ##
    #imp_weights = torch.nn.functional.softmax(logpygivenx + logpx - logqxmgivenxor,0)
    #xms = xmgivenxor.sample().reshape([L,batch_size,p])
    #xm=torch.einsum('ki,kij->ij', imp_weights, xms)
    if miss_x and miss_y:
      ym = torch.mean(ym.reshape([M,-1]),0).reshape([batch_size,1])
    return {'xm': xm, 'ym': ym}
  # return {'xm': xm}
  
  # initialize weights
  def weights_init(layer):
    if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
  
  # Define ADAM optimizer
  if learn_r:
    if not Ignorable:
      params = list(NN_y.parameters()) + list(NN_r.parameters())# + list({'params': mu_x}) + list({'params': scale_x})
    else:
      params = list(NN_y.parameters()) # + list({'params': mu_x}) + list({'params': scale_x})
  else:
    params = list(NN_xm.parameters()) + list(NN_y.parameters())
  if miss_x: params = params + list(NN_xm.parameters())
  if miss_y: params = params + list(NN_ym.parameters())
  optimizer = optim.Adam(params,lr=lr)
  optimizer.add_param_group({"params":mu_x})
  optimizer.add_param_group({"params":scale_x})
  optimizer.add_param_group({"params":alpha})

  # Train and impute every 100 epochs
  mse_train_miss_x=np.array([])
  mse_train_obs_x=np.array([])
  mse_train_miss_y=np.array([])
  mse_train_obs_y=np.array([])
  #mse_pr_epoch = np.array([])
  #CEL_epoch=np.array([]) # Cross-entropy error
  xhat = np.copy(xhat_0) # This will be out imputed data matrix
  yhat = np.copy(yhat_0) # This will be out imputed data matrix

  #trace_ids = np.concatenate([np.where(R[:,0]==0)[0][0:2],np.where(R[:,0]==1)[0][0:2]])
  trace_ids = np.arange(0,10)
  if (trace): print(xhat_0[trace_ids])

  if miss_x: NN_xm.apply(weights_init)
  if miss_y: NN_ym.apply(weights_init)
  NN_y.apply(weights_init)
  if (learn_r and not Ignorable): NN_r.apply(weights_init)
  
  time_train=[]
  time_impute=[]
  LB_epoch=[]
  sum_logpy_epoch =[]
  sum_logqxm_epoch=[]
  sum_logqym_epoch=[]
  sum_logpr_epoch=[]
  sum_logpx_epoch=[]

  # only assign xfull to cuda if it's necessary (save GPU ram)
  if add_miss_term or not draw_miss: cuda_xfull = torch.from_numpy(xfull).float().cuda(); cuda_yfull = torch.from_numpy(yfull).float().cuda()
  else: cuda_xfull = None; cuda_yfull = None

  if train==1:
    # Training+Imputing
    for ep in range(1,n_epochs):
      perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
      batches_xfull = np.array_split(xfull[perm,],n/bs)
      batches_x = np.array_split(xhat_0[perm,], n/bs)
      batches_yfull = np.array_split(yfull[perm,],n/bs)
      batches_y = np.array_split(yhat_0[perm,], n/bs)
      batches_mask_x = np.array_split(mask_x[perm,], n/bs)     # only mask for x. for y --> include as a new mask
      batches_mask_y = np.array_split(mask_y[perm,], n/bs)     # only mask for x. for y --> include as a new mask
      if covars: batches_covar = np.array_split(covars_miss[perm,], n/bs)
      splits = np.array_split(perm,n/bs)
      t0_train=time.time()
      for it in range(len(batches_x)):
        if (add_miss_term or not draw_miss): b_xfull = torch.from_numpy(batches_xfull[it]).float().cuda(); b_yfull = torch.from_numpy(batches_yfull[it]).float().cuda()
        else: b_xfull = None; b_yfull = None
        b_x = torch.from_numpy(batches_x[it]).float().cuda()
        b_y = torch.from_numpy(batches_y[it]).float().cuda()
        b_mask_x = torch.from_numpy(batches_mask_x[it]).float().cuda()
        b_mask_y = torch.from_numpy(batches_mask_y[it]).float().cuda()
        if covars: b_covar = torch.from_numpy(batches_covar[it]).float().cuda()
        else: b_covar = None
        
        optimizer.zero_grad()
        if miss_x: NN_xm.zero_grad()
        if miss_y: NN_ym.zero_grad()
        NN_y.zero_grad()
        if (learn_r and not Ignorable): NN_r.zero_grad()
        #mu_x.zero_grad(); scale_x.zero_grad()

        loss_fit = compute_loss(iota_xfull=b_xfull, iota_yfull=b_yfull, iota_x = b_x, iota_y = b_y, mask_x = b_mask_x, mask_y = b_mask_y, covar_miss = b_covar, temp=temp)
        # inputs: iota_xfull,iota_x,iota_y,mask,covar_miss,temp

        loss=loss_fit['neg_bound']
        loss.backward()
        optimizer.step()
      time_train=np.append(time_train,time.time()-t0_train)

      if covars: torch_covars_miss = torch.from_numpy(covars_miss).float().cuda()
      else: torch_covars_miss = None

      loss_fit=compute_loss(iota_xfull = cuda_xfull, iota_yfull = cuda_yfull, iota_x = torch.from_numpy(xhat_0).float().cuda(), iota_y = torch.from_numpy(yhat_0).float().cuda(), mask_x = torch.from_numpy(mask_x).float().cuda(), mask_y = torch.from_numpy(mask_y).float().cuda(), covar_miss = torch_covars_miss, temp=temp)
      # inputs: iota_xfull,iota_x,iota_y,mask,covar_miss,temp

      LB=(-loss_fit['neg_bound'].cpu().data.numpy())
      LB_epoch=np.append(LB_epoch,LB)
      #learned_probMissing = torch.mean(loss_fit['params_r']['probs'].reshape([M,-1]),axis=0).reshape([n,p]).cpu().data.numpy()
      #mse_pr=np.mean(pow(learned_probMissing[:,0]-pR[:,0],2)) # just the first column (missing column in toy, adjust later)
      #mse_pr_epoch=np.append(mse_pr_epoch, mse_pr)
      #CEL=np.sum(-np.log(learned_probMissing[mask_x==1])) + np.sum(-np.log(1-learned_probMissing[mask_x==0]))
      #CEL_epoch = np.append(CEL_epoch, CEL)
      
      sum_logpy_epoch=np.append(sum_logpy_epoch,loss_fit['sum_logpygivenx'])
      sum_logqxm_epoch=np.append(sum_logqxm_epoch,loss_fit['sum_logqxm'])
      sum_logqym_epoch=np.append(sum_logqym_epoch,loss_fit['sum_logqym'])
      sum_logpr_epoch=np.append(sum_logpr_epoch,loss_fit['sum_logpr'])
      sum_logpx_epoch=np.append(sum_logpx_epoch,loss_fit['sum_logpx'])
      
      if ep % 100 == 1:
        print('Epoch %g' %ep)
        print('Likelihood lower bound  %g' %LB) # Gradient step   

        if trace:
          print("mean, p(x):")
          print(mu_x.cpu().data.numpy())
          print("scale p(x):")
          print(scale_x.cpu().data.numpy())
          if family=="Gaussian":
            print("mean, p(y|x):")    # E[y|x] = beta0 + beta*x
            print(torch.mean(loss_fit['params_y']['mean'].reshape([M,-1]),0).reshape([-1,1])[trace_ids])
            print("scale, p(y|x):")
            print(torch.mean(loss_fit['params_y']['scale'].reshape([M,-1]),0).reshape([-1,1])[trace_ids])
          elif family=="Multinomial":
            print("probs, p(y|x):")
            print(torch.mean(loss_fit['params_y']['probs'].reshape([M,-1]),0).reshape([-1,C])[trace_ids])
          elif family=="Poisson":
            print("lambda, p(y|x):")
            print(torch.mean(loss_fit['params_y']['lambda'].reshape([M,-1]),0).reshape([-1,1])[trace_ids])
          
          if miss_x:
            print("mean (avg over M samples), q(xm|xo,r):")
            print(loss_fit['params_xm']['mean'][trace_ids])
            print("scale (avg over M samples), q(xm|xo,r):")
            print(loss_fit['params_xm']['scale'][trace_ids])
          if miss_y:
            print("mean (avg over M samples), q(ym|yo,r,xm,xo):")
            print(loss_fit['params_ym']['mean'][trace_ids])
            print("scale (avg over M samples), q(ym|yo,r,xm,xo):")
            print(loss_fit['params_ym']['scale'][trace_ids])
          
          if not Ignorable:
            print("prob_Missing (avg over M, then K samples):")
            print(torch.mean(loss_fit['params_r']['probs'].reshape([M,-1]),axis=0).reshape([n,-1])[trace_ids])
        

        t0_impute=time.time()
        batches_xfull = np.array_split(xfull,n/impute_bs)
        batches_x = np.array_split(xhat_0, n/impute_bs)
        batches_y = np.array_split(yfull, n/impute_bs)
        batches_mask_x = np.array_split(mask_x, n/impute_bs)
        batches_mask_y = np.array_split(mask_y, n/impute_bs)
        if covars: batches_covar = np.array_split(covars_miss, n/impute_bs)
        splits = np.array_split(range(n),n/impute_bs)
        for it in range(len(batches_x)):
          if (add_miss_term or not draw_miss): b_xfull = torch.from_numpy(batches_xfull[it]).float().cuda(); b_yfull = torch.from_numpy(batches_yfull[it]).float().cuda()
          else: b_xfull = None; b_yfull=None
          b_x = torch.from_numpy(batches_x[it]).float().cuda()
          b_y = torch.from_numpy(batches_y[it]).float().cuda()
          b_mask_x = torch.from_numpy(batches_mask_x[it]).float().cuda()
          b_mask_y = torch.from_numpy(batches_mask_y[it]).float().cuda()
          if covars: b_covar = torch.from_numpy(batches_covar[it]).float().cuda()
          else: b_covar = None
          impute_fit=impute(iota_xfull = b_xfull, iota_yfull = b_yfull, iota_x = b_x, iota_y = b_y, mask_x = b_mask_x, mask_y = b_mask_y, covar_miss = b_covar, L=L, temp=temp)
          # inputs: iota_xfull,iota_x,iota_y,mask,covar_miss,L,temp

          # imputing xmiss:
          b_xhat = xhat[splits[it],:]
          b_yhat = yhat[splits[it],:]
          #b_xhat[batches_mask_x[it]] = torch.mean(loss_fit['params_x']['mean'].reshape([niw,-1]),axis=0).reshape([n,p])[splits[it],:].cpu().data.numpy()[batches_mask_x[it]] # observed data. nop need to impute
          if miss_x: b_xhat[~batches_mask_x[it]] = impute_fit['xm'].cpu().data.numpy()[~batches_mask_x[it]]       # just missing impute
          if miss_y: b_yhat[~batches_mask_y[it]] = impute_fit['ym'].cpu().data.numpy()[~batches_mask_y[it]]       # just missing impute
          xhat[splits[it],:] = b_xhat
          yhat[splits[it],:] = b_yhat

          # imputing ymiss here? add here:
          ###b_yhat[~batches_mask_y[it]] = impute_fit['xm'].cpu().data.numpy()[~batches_mask_y[it]]

        time_impute=np.append(time_impute,time.time()-t0_impute)

        err_x = mse(xhat,xfull,mask_x)
        err_y = mse(yhat,yfull,mask_y)

        mse_train_miss_x = np.append(mse_train_miss_x,np.array([err_x['miss']]),axis=0)
        mse_train_obs_x = np.append(mse_train_obs_x,np.array([err_x['obs']]),axis=0)
        mse_train_miss_y = np.append(mse_train_miss_y,np.array([err_y['miss']]),axis=0)
        mse_train_obs_y = np.append(mse_train_obs_y,np.array([err_y['obs']]),axis=0)
        
        print('Observed MSE x:  %g' %err_x['obs'])   # these aren't reconstructed/imputed
        print('Missing MSE x:  %g' %err_x['miss'])
        print('Observed MSE y:  %g' %err_y['obs'])   # these aren't reconstructed/imputed
        print('Missing MSE y:  %g' %err_y['miss'])
        print('-----')

    #plt.plot(range(1,n_epochs,100),mse_train_obs,color="blue")
    #plt.title("Imputation MSE (Observed)")
    #plt.xlabel("Epochs")
    #plt.show()
    # plt.plot(range(1,n_epochs,100),mse_train_miss_x,color="blue")
    # plt.title("Imputation MSE (Missing, x)")
    # plt.xlabel("Epochs")
    # plt.show()
    # plt.plot(range(1,n_epochs,100),mse_train_miss_y,color="blue")
    # plt.title("Imputation MSE (Missing, y)")
    # plt.xlabel("Epochs")
    # plt.show()

    plot_first_epoch=1
    #plot_first_epoch=200
    #plt.plot(range(plot_first_epoch,n_epochs),mse_pr_epoch[plot_first_epoch-1:],color="blue")
    #plt.title("MSE of probMissing")
    #plt.xlabel("Epochs")
    #plt.show()
    #plt.plot(range(plot_first_epoch,n_epochs),CEL_epoch[plot_first_epoch-1:],color="green")
    #plt.title("Cross-Entropy Loss (mask_x)")
    #plt.xlabel("Epochs")
    #plt.show()
    # if miss_x:
    #   plt.plot(range(plot_first_epoch,n_epochs),sum_logqxm_epoch[plot_first_epoch-1:],color="blue")
    #   plt.title("log q(x^m|x^o,r)")
    #   plt.xlabel("Epochs")
    #   plt.show()
    # if miss_y:
    #   plt.plot(range(plot_first_epoch,n_epochs),sum_logqym_epoch[plot_first_epoch-1:],color="blue")
    #   plt.title("log q(y^m|y^o,r,x^m,x^o)")
    #   plt.xlabel("Epochs")
    #   plt.show()
    # plt.plot(range(plot_first_epoch,n_epochs),sum_logpr_epoch[plot_first_epoch-1:],color="blue")
    # plt.title("log p(r|x)")
    # plt.xlabel("Epochs")
    # plt.show()
    # plt.plot(range(plot_first_epoch,n_epochs),sum_logpy_epoch[plot_first_epoch-1:],color="blue")
    # plt.title("log p(y|x)")
    # plt.xlabel("Epochs")
    # plt.show()
    # plt.plot(range(plot_first_epoch,n_epochs),sum_logpx_epoch[plot_first_epoch-1:],color="red")
    # plt.title("log p(x)")
    # plt.xlabel("Epochs")
    # plt.show()
    # plt.plot(range(plot_first_epoch,n_epochs),LB_epoch[plot_first_epoch-1:],color="red")
    # plt.title("Lower Bound")
    # plt.xlabel("Epochs")
    # plt.show()

    if (learn_r): saved_model={'NN_xm': NN_xm, 'NN_ym': NN_ym, 'NN_y': NN_y, 'NN_r': NN_r, 'mu_x':mu_x, 'scale_x':scale_x}
    else: saved_model={'NN_xm': NN_xm, 'NN_ym': NN_ym, 'NN_y': NN_y, 'mu_x':mu_x, 'scale_x':scale_x}
    mse_train={'miss_x':mse_train_miss_x,'obs_x':mse_train_obs_x, 'miss_y':mse_train_miss_y,'obs_y':mse_train_obs_y}
    train_params = {'h1':h1, 'h2':h2, 'h3':h3, 'sigma':sigma, 'bs':bs, 'n_epochs':n_epochs, 'lr':lr, 'L':L, 'M':M, 'covars_r_x':covars_r_x, 'covars_r_y':covars_r_y, 'n_hidden_layers':n_hidden_layers, 'n_hidden_layers_r':n_hidden_layers_r, 'pre_impute_value':pre_impute_value}
    all_params = {'x': {'mean':mu_x.cpu().data.numpy(), 'scale':scale_x.cpu().data.numpy()}}
    if family=="Gaussian": all_params['y'] = {'mean':loss_fit['params_y']['mean'].cpu().data.numpy(), 'scale': loss_fit['params_y']['scale'].cpu().data.numpy()}
    elif family=="Multinomial": all_params['y'] =  {'probs':loss_fit['params_y']['probs'].cpu().data.numpy()}
    elif family=="Poisson": all_params['y'] = {'lambda':loss_fit['params_y']['lambda'].cpu().data.numpy()}
    if not Ignorable: all_params['r'] = {'probs': loss_fit['params_r']['probs'].cpu().data.numpy()}

    if miss_x: all_params['xm'] = {'mean':loss_fit['params_xm']['mean'].cpu().data.numpy(),'scale':loss_fit['params_xm']['scale'].cpu().data.numpy()}
    if miss_y: all_params['ym'] = {'mean':loss_fit['params_ym']['mean'].cpu().data.numpy(),'scale':loss_fit['params_ym']['scale'].cpu().data.numpy()}
    return {'train_params':train_params, 'all_params':all_params, 'loss_fit':loss_fit,'impute_fit':impute_fit,'saved_model': saved_model,'LB': LB,'LB_epoch': LB_epoch,'time_train': time_train,'time_impute': time_impute,'MSE': mse_train, 'xhat': xhat, 'yhat':yhat, 'yfull':yfull, 'mask_x': mask_x, 'mask_y':mask_y, 'norm_means_x':norm_means_x, 'norm_sds_x':norm_sds_x,'norm_mean_y':norm_mean_y, 'norm_sd_y':norm_sd_y}
  else:
    # validating (hyperparameter values) or testing
    mu_x = saved_model['mu_x']; scale_x = saved_model['scale_x']
    if (miss_x): NN_xm=saved_model['NN_xm']
    if (miss_y): NN_ym=saved_model['NN_ym']
    NN_y=saved_model['NN_y']
    if (learn_r and not Ignorable): NN_r=saved_model['NN_r']

    if add_miss_term or not draw_miss: cuda_xfull = torch.from_numpy(xfull).float().cuda(); cuda_yfull = torch.from_numpy(yfull).float().cuda()
    else: cuda_xfull = None; cuda_yfull = None

    for ep in range(1,n_epochs):
      if (miss_x): NN_xm.zero_grad()
      if (miss_y): NN_ym.zero_grad()
      NN_y.zero_grad()
      if (learn_r and not Ignorable): NN_r.zero_grad()

      # Validation set is much smaller, so including all observations should be fine?
      if covars: torch_covars_miss = torch.from_numpy(covars_miss).float().cuda()
      else: torch_covars_miss = None
      
      loss_fit=compute_loss(iota_xfull = cuda_xfull, iota_yfull = cuda_yfull, iota_x = torch.from_numpy(xhat_0).float().cuda(), iota_y = torch.from_numpy(yhat_0).float().cuda(), mask_x = torch.from_numpy(mask_x).float().cuda(), mask_y = torch.from_numpy(mask_y).float().cuda(), covar_miss = torch_covars_miss, temp=temp_min)
      # inputs: iota_xfull,iota_x,iota_y,mask,covar_miss,temp

      #LB=(-np.log(K) - np.log(M) - loss_fit['neg_bound'].cpu().data.numpy())  
      LB=(-loss_fit['neg_bound'].cpu().data.numpy())   
      
      t0_impute=time.time()
      impute_fit=impute(iota_xfull = cuda_xfull, iota_yfull = cuda_yfull, iota_x = torch.from_numpy(xhat_0).float().cuda(), iota_y = torch.from_numpy(yhat_0).float().cuda(), mask_x = torch.from_numpy(mask_x).float().cuda(), mask_y = torch.from_numpy(mask_y).float().cuda(), covar_miss = torch_covars_miss,L=L,temp=temp_min)
      # inputs: iota_xfull,iota_x,iota_y,mask_x,covar_miss,L,temp
      time_impute=np.append(time_impute,time.time()-t0_impute)

      # impute xm:
      #xhat[mask_x] = torch.mean(loss_fit['params_x']['mean'].reshape([niw,-1]),axis=0).reshape([n,p]).cpu().data.numpy()[mask]
      if miss_x: xhat[~mask_x] = impute_fit['xm'].cpu().data.numpy()[~mask_x]

      # impute ym:
      if miss_y: yhat[~mask_y] = impute_fit['ym'].cpu().data.numpy()[~mask_y]

      err_x = mse(xhat, xfull, mask_x)
      err_y = mse(yhat, yfull, mask_y)
      
      print('Observed MSE x:  %g' %err_x['obs'])   # these aren't reconstructed/imputed
      print('Missing MSE x:  %g' %err_x['miss'])
      print('Observed MSE y:  %g' %err_y['obs'])   # these aren't reconstructed/imputed
      print('Missing MSE y:  %g' %err_y['miss'])
      print('-----')
      
    mse_test={'miss_x':err_x['miss'],'obs_x':err_x['obs'], 'miss_y':err_y['miss'],'obs_y':err_y['obs']}
    if (learn_r): saved_model={'NN_xm': NN_xm, 'NN_ym': NN_ym, 'NN_y': NN_y, 'NN_r': NN_r, 'mu_x':mu_x, 'scale_x':scale_x}
    else: saved_model={'NN_xm': NN_xm, 'NN_ym':NN_ym, 'NN_y': NN_y, 'mu_x':mu_x, 'scale_x':scale_x}
    all_params = {'x': {'mean':mu_x.cpu().data.numpy(), 'scale':scale_x.cpu().data.numpy()}}
    if family=="Gaussian": all_params['y'] = {'mean':loss_fit['params_y']['mean'].cpu().data.numpy(), 'scale': loss_fit['params_y']['scale'].cpu().data.numpy()}
    elif family=="Multinomial": all_params['y'] =  {'probs':loss_fit['params_y']['probs'].cpu().data.numpy()}
    elif family=="Poisson": all_params['y'] = {'lambda': loss_fit['params_y']['lambda'].cpu().data.numpy()}
    if not Ignorable: all_params['r'] = {'probs': loss_fit['params_r']['probs'].cpu().data.numpy()}

    if miss_x: all_params['xm'] = {'mean':loss_fit['params_xm']['mean'].cpu().data.numpy(),'scale':loss_fit['params_xm']['scale'].cpu().data.numpy()}
    if miss_y: all_params['ym'] = {'mean':loss_fit['params_ym']['mean'].cpu().data.numpy(),'scale':loss_fit['params_ym']['scale'].cpu().data.numpy()}
    
    train_params = {'h1':h1, 'h2':h2, 'h3':h3, 'sigma':sigma, 'bs':bs, 'n_epochs':n_epochs, 'lr':lr, 'L':L, 'M':M, 'covars_r_x':covars_r_x, 'covars_r_y':covars_r_y, 'n_hidden_layers':n_hidden_layers, 'n_hidden_layers_r':n_hidden_layers_r, 'pre_impute_value':pre_impute_value}
    
    # w0 = (NN_y[0].bias).cpu().data.numpy()
    w = (NN_y[0].weight).cpu().data.numpy()
    return {'w':w,'train_params':train_params,'loss_fit':loss_fit,'impute_fit':impute_fit,'saved_model': saved_model,'all_params':all_params,'LB': LB,'time_impute': time_impute,'MSE': mse_test, 'xhat': xhat, 'yhat':yhat, 'xfull': xfull, 'yfull':yfull, 'mask_x': mask_x, 'mask_y':mask_y, 'norm_means_x':norm_means_x, 'norm_sds_x':norm_sds_x,'norm_mean_y':norm_mean_y, 'norm_sd_y':norm_sd_y}
