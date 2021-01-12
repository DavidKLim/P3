# N=100000; P=2; pi=0.5

simulateData = function(N, P, data_types, family, seed,
                        mu=0, sd=1, beta=5, C=NULL){
  # if params are specified, then simulate from distributions
  # if file.name specified, then read in data file --> should have "data"
  # seed determined by sim_index
    
  # mu=0; sd=1; beta=5
  
  X = matrix(nrow=N, ncol=P)
  set.seed(seed)
  # for(p in 1:P){
  #   if(data_types[p]=="real"){
  #     X[, p] = rnorm(N,mean=mu,sd=sd)
  #   } else if(data_types[p]=="cat"){
  #     #### commented out for now; just deal with Gaussian covariates for now...
  #     # X[, p] = apply(rmultinom(N, 1, rep(1/C,C)), 2, which.max)
  #     ### for categorical data, we have to create dummy variables for categories.... (in simulating Y)
  #   } else if(data_types[p]=="count"){
  #     X[, p] = rpois(N, lambda=mu)
  #   }
  # }
  mu = rep(mu, sum(data_types=="real"))
  Sigma=diag(sum(data_types=="real"))
  Sigma[Sigma==0]=0.5
  library(MASS)
  X[, data_types=="real"] = mvrnorm(N,mu,Sigma)
    
  # family="Gaussian" --> Gaussian data for Y
  if(family=="Gaussian"){
    # Simulate y from X --> y = Xb + e
    beta0s = 0
    # betas = rnorm(P)   # sampled coefs (may be bad. if 0 --> no relationship between X and Y)
    betas = sample(c(-1*beta,beta), P, replace=T)   # -2 or 2 fixed coefs
    # e = rnorm(N,0,1)
    # Y = beta0s + X %*% betas + e
    Y = beta0s + X %*% betas
  } else if(family=="Multinomial"){
    # beta0s = rnorm(C, 0, 1)
    # betas = cbind(rnorm(C, -1, 1), rnorm(C, 1, 1))   # C x P matrix: each covariates' effects on each class
    beta0s = 0
    # betas = matrix(rnorm(C*P),nrow=C,ncol=P)
    betas = matrix(sample(c(-1*beta,beta), C*P, replace=T),nrow=C,ncol=P)
    prs = exp(matrix(beta0s,nrow=N,ncol=C,byrow=T) + X %*% t(betas))
    prs = prs/rowSums(prs)
    Y = apply(prs, 1, sample, x=c(1:C), size=1, replace=F)
  } else if(family=="Poisson"){
    beta0s = 8
    # betas = rnorm(P)  # effect sizes -1 and 1 test case. Must be of length P
    betas = sample(c(-1*beta,beta), P, replace=T)
    Y = round(exp(beta0s + X %*% betas),0)   # log(Y) = eta. Round Y to integer (to simulate count)
  }
  Y = matrix(Y,ncol=1)
  hist(Y)

  data = list(X=X, Y=Y)
  params = list(beta0s=beta0s, betas=betas,
                mu=mu,sd=sd,beta=beta)
  
  return(list(data=data, params=params))
}

simulateMask = function(data, scheme, mechanism, pi, phis, miss_cols, ref_cols, seed){
  
  # mechanism of missingness
  # pi: % of missingness
  # miss_cols: columns to induce missingness on
  # ref_cols: (FOR MAR) columns that are fully observed used as covariate
  # seed determined by sim_index
  n=nrow(data)
  # function s.t. expected proportion of nonmissing (Missing=1) is p. let p=1-p_miss
  find_int = function(p,beta) {
    # Define a path through parameter space
    f = function(t){
      sapply(t, function(y) mean(1 / (1 + exp(-y -x %*% phi))))
    }
    alpha <- uniroot(function(t) f(t) - p, c(-1e6, 1e6), tol = .Machine$double.eps^0.5)$root
    return(alpha)
  }
  inv_logit = function(x){
    return(1/(1+exp(-x)))
  }
  logit = function(x){
    if(x<=0 | x>=1){stop('x must be in (0,1)')}
    return(log(x/(1-x)))
  }
  
  Missing = matrix(1,nrow=nrow(data),ncol=ncol(data))
  prob_Missing = matrix(1,nrow=nrow(data),ncol=ncol(data))
  params=list()

  set.seed(seed)  # for reproducibility
  
  for(j in 1:length(miss_cols)){
    # specifying missingness model covariates
    if(mechanism=="MCAR"){
      x <- matrix(rep(0,n),ncol=1) # for MCAR: no covariate (same as having 0 for all samples)
      phi=0                       # for MCAR: no effect of covariates on missingness (x is 0 so phi doesn't matter)
    }else if(mechanism=="MAR"){
      if(scheme=="UV"){
        x <- matrix(data[,ref_cols[j]],ncol=1)             # missingness dep on just the corresponding ref column (randomly paired)
        phi=phis[ref_cols[j]]
      }else if(scheme=="MV"){
        # check if missing column in ref. col: this would be MNAR (stop computation)
        if(any(ref_cols %in% miss_cols)){stop(sprintf("missing cols in reference. is this intended? this is MNAR not MAR."))}
        x <- matrix(data[,ref_cols],ncol=length(ref_cols)) # missingness dep on all ref columns
        phi=phis[ref_cols]
      }
    }else if(mechanism=="MNAR"){
        # Selection Model
        if(scheme=="UV"){
          # MISSINGNESS OF EACH MISS COL IS ITS OWN PREDICTOR
          x <- matrix(data[,miss_cols[j]],ncol=1) # just the corresponding ref column
          phi=phis[miss_cols[j]]
        }else if(scheme=="MV"){
          # check if missing column not in ref col. this might be MAR if missingness not dep on any other missing data
          if(all(!(ref_cols %in% miss_cols))){warning(sprintf("no missing cols in reference. is this intended? this might be MAR not MNAR"))}
          x <- matrix(data[,ref_cols],ncol=length(ref_cols)) # all ref columns
          phi=phis[ref_cols]         # in MNAR --> ref_cols can overlap with miss_cols (dependent on missingness)
          
          # address when miss_cols/ref_cols/phis are not null (i.e. want to induce missingness on col 2 & 5 based on cols 1, 3, & 4)
        }
    }
    alph <- sapply(pi, function(y) find_int(y,phi))
    mod = alph + x%*%phi
    
    # Pr(Missing_i = 1)
    probs = inv_logit(mod)
    prob_Missing[,miss_cols[j]] = probs
    
    # set seed for each column for reproducibility, but still different across columns
    Missing[,miss_cols[j]] = rbinom(n,1,probs)
    
    params[[j]]=list(phi0=alph, phi=phi, miss=miss_cols[j], ref=ref_cols[j], scheme=scheme)
    print(c(mean(data[Missing[,miss_cols[j]]==0,miss_cols[j]]), mean(data[Missing[,miss_cols[j]]==1,miss_cols[j]])))
  }
  
  return(list(Missing=Missing,
              probs=prob_Missing,
              params=params))
}

# data.file.name=NULL; mask.file.name=NULL; sim.params = list(N=1e5, P=8, data_types=rep("real",P), family="Gaussian", sim_index=1); miss.params = list(scheme="UV", mechanism="MNAR", pi=0.5, phi0=5, miss_cols=NULL, ref_cols=NULL, sim_index=1); case="x"
prepareData = function(data.file.name = NULL, mask.file.name=NULL,
                       sim.params = list(N=1e5, P=8, data_types=NA, family="Gaussian", sim_index=1, ratios=c(train=.6,valid=.2,test=.2), mu=0, sd=1, beta=5, C=NULL),
                       miss.params = list(scheme="UV", mechanism="MNAR", pi=0.5, phi0=5, miss_cols=NULL, ref_cols=NULL),
                       case=c("x","y","xy")){
  print(sim.params)
  # data (simulate or read)
  if(!is.null(data.file.name)){
    # read existing data. list object "data" should be there, with entries "X" and "Y"
    load(data.file.name)
    dataset = unlist(strsplit(data.file.name,"[.]"))[1]   # remove extension
  }else{
    P=sim.params$P; N=sim.params$N
    if(all(is.na(sim.params$data_types))){sim.params$data_types = rep("real",sim.params$P)}
    
    sim.data = simulateData(N=sim.params$N, P=sim.params$P,
                            data_types=sim.params$data_types,
                            family=sim.params$family,
                            seed=sim.params$sim_index*9,
                            mu=sim.params$mu, sd=sim.params$sd,
                            beta=sim.params$beta, C=sim.params$C)
    params = sim.data$params
    dataset = sprintf("Xmean%dsd%d_beta%d_pi%d/SIM_N%d_P%d_X%s_Y%s", params$mu[1], params$sd[1], params$beta, miss.params$pi*100,
                      sim.params$N, sim.params$P, sim.params$data_types[1], sim.params$family)
    
    # family = "Gaussian", "Multinomial", or "Poisson"
    data = sim.data$data
    X=data$X; Y=data$Y
  }
  
  # mask (simulate or read)
  if(!is.null(mask.file.name)){
    # read existing mask. object "mask" should be there
    load(mask.file.name)
    phis=NULL; miss_cols=NULL; ref_cols
    mechanism = NA; pi=NA
  }else{
    mechanism = miss.params$mechanism; pi = miss.params$pi
    set.seed( sim.params$sim_index*9 )
    # phis = c(0, rlnorm(P, log(miss.params$phi0), 0.2)) # default: 0 for intercept, draw values of coefs from log-norm(5, 0.2)
    phis = rlnorm(P, log(miss.params$phi0), 0.2) # SHOULDN"T CONTAIN INTERCEPT... intercept found by critical point in E[R] ~ pi
    # if no reference/miss cols provided --> randomly select 50% each
    if(is.null(miss.params$miss_cols) & is.null(miss.params$ref_cols)){
      miss_cols = sample(1:P, floor(P/2), F); ref_cols = c(1:P)[-miss_cols]
    }else if(!is.null(miss.params$miss_cols) & !is.null(miss.params$ref_cols)){
      miss_cols = miss.params$miss_cols; ref_cols = miss.params$ref_cols
    }else if(is.null(miss.params$miss_cols)){
      ref_cols = miss.params$ref_cols; miss_cols = c(1:P)[-ref_cols]
    }else if(is.null(miss.params$ref_cols)){
      miss_cols = miss.params$miss_cols; ref_cols = c(1:P)[-miss_cols]
    }
    
    if(case=="x"){
      data=X
      ref_cols0=ref_cols; miss_cols0=miss_cols
    }else if(case=="y"){
      data=data.frame(cbind(X[,ref_cols[1]],Y))
      ref_cols0=1; miss_cols0=2
    }else if(case=="xy"){
      data=data.frame(cbind(X,Y))
      miss_cols0=c(miss_cols,  ncol(data))  # add Y to the columns where missingness is imposed
      ref_cols0 = ref_cols
    }
    sim.mask = simulateMask(data=data, scheme=miss.params$scheme, mechanism=miss.params$mechanism,
                            pi=miss.params$pi, phis=phis, miss_cols=miss_cols0, ref_cols=ref_cols0,
                            seed = sim.params$sim_index*9)
    mask = sim.mask$Missing    # 1 observed, 0 missing
    if(case=="x"){
      mask_x = mask; mask_y = rep(1, N)
    }else if(case=="y"){
      mask_x = matrix(1, nrow=N, ncol=P); mask_y = mask
    }else if(case=="xy"){
      mask_x = mask[,1:P]; mask_y = mask[,(P+1)]
    }
  }
  data_type_x = sim.params$data_types[1]; data_type_y = if(family=="Gaussian"){"real"}else if(family=="Multinomial"){"cat"}else if(family=="Poisson"){"cts"}
  
  dir_name = sprintf("Results_X%s_Y%s/%s/miss_%s/phi%d/sim%d", data_type_x, data_type_y, dataset, case, miss.params$phi0, sim.params$sim_index)
  ifelse(!dir.exists(dir_name), dir.create(dir_name,recursive=T), F)
  
  diag_dir_name = sprintf("%s/Diagnostics", dir_name)
  ifelse(!dir.exists(diag_dir_name), dir.create(diag_dir_name,recursive=T), F)
  
  if(grepl("x",case)){
    # plot missing X vals (1 col)
    png(sprintf("%s/%s_X.png",diag_dir_name, mechanism))
    boxplot(X[mask_x[,miss_cols[1]]==0,miss_cols[1]],
            X[mask_x[,miss_cols[1]]==1,miss_cols[1]],
            X[mask_x[,miss_cols[1]]==0,ref_cols[1]], 
            X[mask_x[,miss_cols[1]]==1,ref_cols[1]], 
            names =c("Masked, miss_col", "Unmasked, miss_col", "Masked, ref_col", "Unmasked, ref_col"),
            main="Diagnostics: Values of 1st miss_col and 1st ref_col", sub="Stratified by mask in 1st miss_col",outline=F)
    dev.off()
  }
  if(grepl("y",case)){
    png(sprintf("%s/%s_Y.png",diag_dir_name, mechanism))
    # plot missing Y vals
    boxplot(Y[mask_y==0,],
            Y[mask_y==1,],
            X[mask_y==0,ref_cols[1]], 
            X[mask_y==1,ref_cols[1]], 
            names =c("Masked, Y", "Unmasked, Y", "Masked, ref_col", "Unmasked, ref_col"),
            main="Diagnostics: Values of Y and 1st ref_col (in X)", sub="Stratified by mask in Y",outline=F)
    dev.off()
    
  }
  
  # save: X, Y, mask_x, mask_y
  # save extra params: 
  
  miss.params$miss_cols = miss_cols; miss.params$ref_cols = ref_cols; miss.params$phis = phis
  
  g = sample(cut(
    seq(sim.params$N), 
    sim.params$N*cumsum(c(0,sim.params$ratios)),
    labels = names(sim.params$ratios)
  ))
  
  sim.data$data = NULL; sim.mask$Missing = NULL # these have already been extracted. don't save to params file
  
  save(list=c("X","Y","mask_x","mask_y","g"), file = sprintf("%s/data_%s_%d.RData", dir_name, mechanism, pi*100))
  save(list=c("sim.params","miss.params","sim.data","sim.mask"), file = sprintf("%s/params_%s_%d.RData", dir_name, mechanism, pi*100))
}

phi0=100; pi=0.5; sim_index=1
mu=5; sd=5; beta=5
mechanisms="MNAR"
case="x"

family="Gaussian"; C=NULL
# family="Multinomial"; C=3
for(i in sim_index){
  for(m in 1:length(mechanisms)){
    prepareData(sim.params = list(N=1e5, P=8, data_types=NA, family=family, sim_index=i, ratios=c(train=.6,valid=.2,test=.2), mu=mu, sd=sd, beta=beta, C=C),
                miss.params=list(scheme="UV", mechanism=mechanisms[m], pi=pi, phi0=phi0, miss_cols=NULL, ref_cols=NULL), case=case)
  }
}

