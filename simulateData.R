inv_logit = function(x){
  return(1/(1+exp(-x)))
}

# N=100000; P=2; pi=0.5
# packages: MASS, qrnn
simulateData = function(N, D, P, data_types, family, seed, NL_x, NL_y,
                        beta=0.25, C=NULL, Cy=1){
  # if params are specified, then simulate from distributions
  # if file.name specified, then read in data file --> should have "data"
  # seed determined by sim_index
    
  # mu=0; sd=1; beta=5
  library(qrnn)  # for softplus/sigmoid/elu function
  logsumexp <- function (x) {
    y = max(x)
    y + log(sum(exp(x - y)))
  }
  softmax <- function (x) {
    exp(x - logsumexp(x))
  }
  
  X = matrix(nrow=N, ncol=P)
  set.seed(seed)
  
  P_real=sum(data_types=="real"); P_cat=sum(data_types=="cat"); P_count=sum(data_types=="count"); P_pos=sum(data_types=="pos")
  ## Using SimMultiCorrData
  # library(SimMultiCorrData)
  # rho = diag(P)
  # rho[rho==0] = 0.5
  # x=SimMultiCorrData::rcorrvar(method="Polynomial",n=N,
  #                              k_cont=P_real, k_cat=P_cat, k_pois=P_count,
  #                              #### continuous ####
  #                              means=mus, vars=sds, skews=rep(0, P_real),
  #                              skurts=rep(0, P_real), fifths=rep(0, P_real), sixths=rep(0, P_real),
  #                              #### count ####
  #                              lam = exp( lambdas ),   # exp mean
  #                              # pois_eps = rep(0.0001, P_count),
  #                              #### cat ####
  #                              marginal=lapply(probs,function(y)cumsum(y)[-length(y)]),
  #                              support=lapply(probs,function(y)c(1:length(y))),
  #                              rho=rho,
  #                              seed=999)
  # 
  # X = list( x$continuous_variables, x$ordinal_variables, x$Poisson_variables )
  # X <- matrix(unlist(X), ncol = P, byrow = F)
  
  ## Using Linear/Nonlinear like P2
  X = matrix(nrow=N, ncol=P)
  Ws = list(); Bs = list()  # save these weights/biases in generation of X from Z
  if(NL_x){
    
    # N=100000; P=25; seed=9*sim_index
    set.seed(seed)
    sd1 = 1
    

    ## skew-normal X2
    Z1 = MASS::mvrnorm(N, rep(0,D), Sigma=diag(D))
    Z2 = MASS::mvrnorm(N, rep(0,D), Sigma=diag(D))
    
    generate_nonlinear_data = function(Z,type=c("real","cat","count","pos"), P_type, C=NULL, seed){
      ### tests:
      # type="real"; P_type=5; C=NULL
      # type="cat"; P_type=5; C=3
      # type="count"; P_type=5; C=NULL
      # type="pos"; P_type=5; C=NULL
      ############
      N = nrow(Z); D = ncol(Z)
      if(type %in% c("real", "count","pos")){P0 = P_type} else if(type =="cat"){ P0 = C*P_type}
      W1 = matrix(runif(D*floor(P0/2),0,1),nrow=D,ncol=floor(P0/2)) # weights
      W2 = matrix(runif(D*(P0-floor(P0/2)),0,1),nrow=D,ncol=P0-floor(P0/2)) # weights      # B = matrix(rnorm(N*P,0,0.25),nrow=N,ncol=P_type)
      
      # eta = Z %*% W + B
      eta1 = 10*qrnn::sigmoid(Z2%*%W1)
      eta2 = 5*(cos(Z1%*%W2) + qrnn::sigmoid(Z2%*%W2))
      if(type=="real"){
        X1 = matrix(rnorm(N*floor(P_type/2), eta1, sd = 0.25), nrow=N, ncol=floor(P_type/2))
        X2 = matrix(rnorm(N*(P_type-ncol(X1)), eta2, sd = 0.25), nrow=N, ncol=P_type-ncol(X1))
      }else if(type=="cat"){
        X = matrix(nrow=N,ncol=0)
        for(j in 1:P_type){
          eta0 = eta[,(C*(j-1)+1):(C*j)]  # subset relevant C columns of linear transformation from Z
          probs = t(apply(eta0, 1, function(x) softmax(x)))
          classes = apply(eta0, 1, function(x){which.max(rmultinom(1,1,softmax(x)))}) # eta0 --> softmax probs --> multinom --> designate class
          X = cbind(X, classes)
        }
      }else if(type=="count"){
        X = matrix(rpois(N*P_type, eta+100), N, P_type)  # add constant of 100 to lambda --> larger counts
      }else if(type=="pos"){
        X = matrix(rlnorm(N*P_type, eta, 0.1), N, P_type) # log-sd changed to have range of about 0 - 80
      }
      
    }
    
    # W1 = matrix(runif(D*floor(P/2),0,1),nrow=D,ncol=floor(P/2)) # weights
    # W2 = matrix(runif(D*(P-floor(P/2)),0,1),nrow=D,ncol=P-floor(P/2)) # weights
    # 
    # X1 = matrix(rnorm(N*floor(P/2), 10*qrnn::sigmoid(Z2%*%W1), sd = 0.25), nrow=N, ncol=floor(P/2))
    # X2 = matrix(rnorm(N*(P-ncol(X1)), 5*(cos(Z1%*%W2) + qrnn::sigmoid(Z2%*%W2)), sd = 0.25), nrow=N, ncol=P-ncol(X1))
    # 
    # X=cbind(X1,X2)
    
    if(P_real>0){ X[,data_types=="real"] = generate_nonlinear_data(Z=Z, type="real", P_type=P_real, C=NULL, seed=seed) }
    if(P_cat>0){ X[,data_types=="cat"] = generate_nonlinear_data(Z=Z, type="cat", P_type=P_cat, C=NULL, seed=seed) }
    if(P_count>0){ X[,data_types=="count"] = generate_nonlinear_data(Z=Z, type="count", P_type=P_count, C=NULL, seed=seed) }
    if(P_pos>0){ X[,data_types=="pos"] = generate_nonlinear_data(Z=Z, type="pos", P_type=P_pos, C=NULL, seed=seed) }
    
  }else{
    set.seed(seed)
    Z=matrix(nrow=N,ncol=D)
    for(d in 1:D){
      Z[,d]=rnorm(N,mean=0,sd=1)
    }
    
    generate_linear_data = function(Z,type=c("real","cat","count","pos"), P_type, C=NULL, seed){
      # ### tests:
      # type="real"; P_type=5; C=NULL
      # type="cat"; P_type=1; C=3
      # type="count"; P_type=5; C=NULL
      # type="pos"; P_type=5; C=NULL
      # ############
      N = nrow(Z); D = ncol(Z)
      if(type %in% c("real", "count","pos")){
        P0 = P_type
      } else if(type =="cat"){
        P0 = C*P_type
      }
      W = if(type=="cat"){
        # matrix(3*c(-1,-1,-1,
        #            1,1,1),nrow=D,ncol=P0)
        matrix(3*sample(c(-1,1),size=D*P0,replace=T),nrow=D,ncol=P0)
        ## ord:
        # matrix(runif(D*P0,0,1),nrow=D,ncol=P0) 
      }else{
        # matrix(sample(c(-1,1),D*P0,replace=T)*runif(D*P0,0.5,2),nrow=D,ncol=P0)  # unif (1,2) and +/- random
        # matrix(sample(c(-1,1),D*P0,replace=T)*runif(D*P0,0.5,1),nrow=D,ncol=P0)  # unif (1,2) and +/- random
        # matrix(rnorm(D*P0,0,1),nrow=D,ncol=P0) 
        matrix(rnorm(D*P0,0,0.5),nrow=D,ncol=P0)
      }
      
      # B = matrix(rnorm(N*P0,0,0.25),nrow=N,ncol=P0)
      B = matrix(rnorm(N*P0,0,1),nrow=N,ncol=P0)
      # B0 = 0 #location shift?
      B0 = 2
      
      eta = Z %*% W + B
      eta = apply(eta,2,function(x){(x-mean(x))/sd(x)})  # pre-normalize
      eta = eta + B0  # add location shift
      
      # eta = Z %*% W
      if(type=="real"){
        # X = matrix(rnorm(N*P_type, eta, 0.25), N, P_type)
        X = eta  # if B matrix (noise) is applied to eta
        # X = apply(eta, 2, function(x) (x-mean(x))/sd(x))    # make it mean 0 sd 1
      }else if(type=="cat"){
        X = matrix(nrow=N,ncol=0)
        for(j in 1:P_type){
          eta0 = eta[,(C*(j-1)+1):(C*j)]  # subset relevant C columns of linear transformation from Z
          probs = t(apply(eta0, 1, function(x) softmax(x)))
          classes = apply(eta0, 1, function(x){which.max(rmultinom(1,1,softmax(x)))}) # eta0 --> softmax probs --> multinom --> designate class
          X = cbind(X, classes)
        }
        
        ## ORD SIM (mimicking ordsample() from GenOrd package)
        # Sigma = diag(P_type); Sigma[Sigma==0] = 0.5
        # # eig=eigen(Sigma,symmetric=T)
        # # sqrteigval = diag(sqrt(eig$values),nrow=nrow(Sigma),ncol=ncol(Sigma))
        # # eigvec <- eig$vectors
        # # fry <- eigvec %*% sqrteigval
        # # 
        # # X <- matrix(rnorm(P_type * N), N)
        # # X <- scale(X, TRUE, FALSE)
        # # X <- X %*% svd(X, nu = 0)$v
        # # X <- scale(X, FALSE, TRUE)
        # # X <- fry %*% t(X)
        # # X <- t(X)        ## https://github.com/AFialkowski/SimMultiCorrData/blob/master/R/rcorrvar.R see Y_cat?
        # 
        # marginal=list(); support=list()
        # for(i in 1:P_type){
        #   marginal[[i]] = c(1:(C-1))/C
        #   support[[i]] = 1:C   # can be different Cs across variables, but keep it constant for now
        # }
        # # draw = matrix(MASS::mvrnorm(N*P_type, rep(0,P_type), Sigma), nrow=N, ncol=P_type)
        # # draw = matrix(rnorm(N*P_type, eta, Sigma), nrow=N, ncol=P_type)
        # draw = eta
        # X = matrix(nrow=N,ncol=P_type)
        # for(i in 1:P_type){
        #   X[,i] = as.integer(cut(draw[,i], breaks=c(min(draw[,i])-1, qnorm(marginal[[i]]), max(draw[,i])+1)))
        #   X[,i] = support[[i]][X[,i]]
        # }
        
      }else if(type=="count"){
        X = matrix(rpois(N*P_type, eta+100), N, P_type)  # add constant of 100 to lambda --> larger counts
      }else if(type=="pos"){
        X = matrix(rlnorm(N*P_type, eta, 0.1), N, P_type) # log-sd changed to have range of about 0 - 80
      }
      return(list(X=X,W=W,B=B))
    }
    
    if(P_real>0){
      fit = generate_linear_data(Z=Z, type="real", P_type=P_real, C=NULL, seed=seed)
      X[,data_types=="real"] = fit$X; Ws$"real" = fit$W; Bs$"real" = fit$B }
    if(P_cat>0){
      fit = generate_linear_data(Z=Z, type="cat", P_type=P_cat, C=C, seed=seed)
      X[,data_types=="cat"] = fit$X; Ws$"cat" = fit$W; Bs$"cat" = fit$B }
    if(P_count>0){ 
      fit = generate_linear_data(Z=Z, type="count", P_type=P_count, C=NULL, seed=seed)
      X[,data_types=="count"] = fit$X; Ws$"count" = fit$W; Bs$"count" = fit$B}
    if(P_pos>0){ 
      fit = generate_linear_data(Z=Z, type="pos", P_type=P_pos, C=NULL, seed=seed)
      X[,data_types=="pos"] = fit$X; Ws$"pos" = fit$W; Bs$"pos" = fit$B}
    
  }

  
  if(family=="Multinomial"){ 
    beta1 = matrix(0, ncol=1, nrow=P_real+P_count+P_cat)  # class 1: reference. effects of X on Pr(Y) are 0
    betas_real = matrix(sample(c(-1*beta, beta), P_real*(Cy-1), replace=T),nrow=P_real,ncol=Cy-1)
    # (most proper:  for cat vars: diff effect per level. not doing this right now --> same effect from 1 --> 2, 2 --> 3, etc.)
    # betas_count = sample(c(-beta/600, beta/600), P_count, replace=T)      # effect is twice the magnitude of the effect of continuous values?
    # betas_cat = sample(c(-2*beta, 2*beta), P_cat, replace=T)       # mean(cts) = 5, mean(cat) = 2, mean(count) = exp(8) ~ 2981
    betas_count = matrix(sample(c(-1*beta, beta), P_count*(Cy-1), replace=T),nrow=P_count,ncol=Cy-1)      # effect is twice the magnitude of the effect of continuous values?
    
    betas_cat = matrix(sample(c(-2*beta, 2*beta), P_cat*(Cy-1), replace=T), nrow=P_cat, ncol=Cy-1)
    betas = cbind(beta1, rbind(betas_real, betas_cat, betas_count))
    epsilon = matrix(rnorm(N*(Cy-1), 0, 0.1), nrow=N,ncol=Cy)
    
  } else{ 
    betas_real = sample(c(-1*beta, beta), P_real, replace=T)
    # (most proper:  for cat vars: diff effect per level. not doing this right now --> same effect from 1 --> 2, 2 --> 3, etc.)
    # betas_count = sample(c(-beta/600, beta/600), P_count, replace=T)      # effect is twice the magnitude of the effect of continuous values?
    # betas_cat = sample(c(-2*beta, 2*beta), P_cat, replace=T)       # mean(cts) = 5, mean(cat) = 2, mean(count) = exp(8) ~ 2981
    betas_count = sample(c(-1*beta, beta), P_count, replace=T)      # effect is twice the magnitude of the effect of continuous values?
    betas_cat = sample(c(-2*beta, 2*beta), P_cat, replace=T)
    betas = c(betas_real, betas_cat, betas_count)
    epsilon = rnorm(N, 0, 0.1)
    
  } # mean(cts) = 5, mean(cat) = 2, mean(count) = exp(8) ~ 2981
  
  # epsilon = 0
  if(NL_y){
    
  }else{
    # family="Gaussian" --> Gaussian data for Y
    if(family=="Gaussian"){
      # Simulate y from X --> y = Xb + e
      beta0s = 0
      eta = beta0s + X %*% betas + epsilon
      # betas = sample(c(-1*beta,beta), P, replace=T)   # defined together in mixed data types
      Y = eta
      prs = NA
    } else if(family=="Multinomial"){
      beta0s = 0
      eta = beta0s + X %*% betas + epsilon
      # betas = matrix(sample(c(-1*beta,beta), C*P, replace=T),nrow=C,ncol=P)    # defined together in mixed data types
      # prs = exp(matrix(beta0s,nrow=N,ncol=C,byrow=T) + X %*% betas)
      prs = t(apply(eta,1,softmax))
      # prs = exp(beta0s + X %*% betas + epsilon)
      # prs = prs/rowSums(prs)
      # Y = apply(prs, 1, sample, x=c(1:Cy), size=1, replace=F)
      Y = apply(prs, 1, function(x) which.max(rmultinom(1, 1, x)))  # categorical distribution
      
      # Y = rbinom(N,1,prs)  # for binary outcome only
    } else if(family=="Poisson"){
      beta0s = 8
      eta = beta0s + X %*% betas + epsilon
      # betas = sample(c(-1*beta,beta), P, replace=T)    # defined together in mixed data types
      Y = round(exp(eta),0)   # log(Y) = eta. Round Y to integer (to simulate count)
      prs = NA
    }
    Y = matrix(Y,ncol=1)
  }
  
  # hist(Y)
  
  # X[,data_types=="count"] = exp(X[,data_types=="count"])    # undo log transf of X

  data = list(X=X, Y=Y)
  params = list(beta0s=beta0s, betas=betas, beta=beta, prs = prs, Ws=Ws, Bs=Bs)
  
  return(list(data=data, params=params))
}

simulateMask = function(data, scheme, mechanism, NL_r, pi, phis, miss_cols, ref_cols, seed){
  
  # mechanism of missingness
  # 1 - pi: % of missingness
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
        covar = if(NL_r){ log(data[,ref_cols[j]] + abs(data[,ref_cols[j]]) + 0.01) }else{ data[,ref_cols[j]] }
        x <- matrix(covar,ncol=1)             # missingness dep on just the corresponding ref column (randomly paired)
        phi=phis[ref_cols[j]]
      }else if(scheme=="MV"){
        # check if missing column in ref. col: this would be MNAR (stop computation)
        if(any(ref_cols %in% miss_cols)){stop(sprintf("missing cols in reference. is this intended? this is MNAR not MAR."))}
        covars = if(NL_r){ apply(data[,ref_cols], 2, function(x){log(x+abs(x)+0.01)}) }else{ data[,ref_cols] }
        x <- matrix(covars,ncol=length(ref_cols)) # missingness dep on all ref columns
        phi=phis[ref_cols]
      }
    }else if(mechanism=="MNAR"){
        # Selection Model
        if(scheme=="UV"){
          # MISSINGNESS OF EACH MISS COL IS ITS OWN PREDICTOR
          covar = if(NL_r){ log(data[,miss_cols[j]] + abs(data[,miss_cols[j]]) + 0.01) }else{ data[,miss_cols[j]] }
          x <- matrix(covar,ncol=1) # just the corresponding ref column
          phi=phis[miss_cols[j]]
        }else if(scheme=="MV"){
          # check if missing column not in ref col. this might be MAR if missingness not dep on any other missing data
          if(all(!(ref_cols %in% miss_cols))){warning(sprintf("no missing cols in reference. is this intended? this might be MAR not MNAR"))}
          covars = if(NL_r){ apply(data[,ref_cols], 2, function(x){log(x+abs(x)+0.01)}) }else{ data[,ref_cols] }
          x <- matrix(data[,ref_cols],ncol=length(ref_cols)) # all ref columns
          phi=phis[ref_cols]         # in MNAR --> ref_cols can overlap with miss_cols (dependent on missingness)
          
          # address when miss_cols/ref_cols/phis are not null (i.e. want to induce missingness on col 2 & 5 based on cols 1, 3, & 4)
        }
    }
    alph <- sapply(1-pi, function(y) find_int(y,phi))
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
                       sim.params = list(N=1e5, D=2, P=8, data_types=NA, family="Gaussian", sim_index=1, ratios=c(train=.6,valid=.2,test=.2),
                                         beta=.25, C=NULL, Cy=NULL, NL_x=F, NL_y=F),
                       miss.params = list(scheme="UV", mechanism="MNAR", pi=0.5, phi0=5, miss_pct_features=50, miss_cols=NULL, ref_cols=NULL, NL_r=F),
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
    
    # N=sim.params$N; P=sim.params$P
    # data_types=sim.params$data_types
    # family=sim.params$family
    # seed=sim.params$sim_index*9
    # mus=sim.params$mus; sds=sim.params$sds
    # lambdas=sim.params$lambdas; probs=sim.params$probs
    # beta=sim.params$beta; C=sim.params$C
    
    sim.data = simulateData(N=sim.params$N, D=sim.params$D, P=sim.params$P,
                            data_types=sim.params$data_types,
                            family=sim.params$family, NL_x=sim.params$NL_x, NL_y=sim.params$NL_y,
                            seed=sim.params$sim_index*9,
                            beta=sim.params$beta, C=sim.params$C, Cy=sim.params$Cy)
    params = sim.data$params
    dataset = sprintf("Xr%dct%dcat%d_beta%f_pi%d/SIM_N%d_P%d_D%d",sum(sim.params$data_types=="real"),
                      sum(sim.params$data_types=="count"),sum(sim.params$data_types=="cat"), params$beta, miss.params$pi*100,
                      sim.params$N, sim.params$P, sim.params$D)
    
    # family = "Gaussian", "Multinomial", or "Poisson"
    data = sim.data$data
    X=data$X; Y=data$Y
  }
  
  print("Data simulated")
  miss_pct_features = miss.params$miss_pct_features
  
  # mask (simulate or read)
  if(!is.null(mask.file.name)){
    # read existing mask. object "mask" should be there
    load(mask.file.name)
    phis=NULL; miss_cols=NULL; ref_cols=NULL
    mechanism = NA; pi=NA
  }else{
    mechanism = miss.params$mechanism; pi = miss.params$pi
    set.seed( sim.params$sim_index*9 )
    # phis = c(0, rlnorm(P, log(miss.params$phi0), 0.2)) # default: 0 for intercept, draw values of coefs from log-norm(5, 0.2)
    phis = rlnorm(P, log(miss.params$phi0), 0.2) # SHOULDN"T CONTAIN INTERCEPT... intercept found by critical point in E[R] ~ pi
    if(any(sim.params$data_types=="cat")){ phis[sim.params$data_types=="cat"] = phis[sim.params$data_types=="cat"] * 2 } # categorical effects on missingness is 1/5th of real effects
    # if no reference/miss cols provided --> randomly select 50% each
    if(is.null(miss.params$miss_cols) & is.null(miss.params$ref_cols)){
      # miss_cols = sample(1:P, floor(P/2), F); ref_cols = c(1:P)[-miss_cols]
      miss_pct_features = miss.params$miss_pct_features
      miss_cols = sample(1:P, ceiling(P*miss_pct_features/100), F)
      ref_cols = c(1:P)[-miss_cols]
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
    sim.mask = simulateMask(data=data, scheme=miss.params$scheme, mechanism=miss.params$mechanism, NL_r=miss.params$NL_r,
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
  
  print("Mask simulated")
  data_type_x = if(all(sim.params$data_types==sim.params$data_types[1])){ sim.params$data_types[1] } else{ "mixed" }
  data_type_y = if(sim.params$family=="Gaussian"){"real"}else if(sim.params$family=="Multinomial"){"cat"}else if(sim.params$family=="Poisson"){"cts"}
  
  iNL_x = if(sim.params$NL_x){"NL"}else{""}
  iNL_y = if(sim.params$NL_y){"NL"}else{""}
  iNL_r = if(miss.params$NL_r){"NL_"}else{""}
  dir_name = sprintf("Results_%sX%s_%sY%s/%s/miss_%s%s/phi%d/sim%d", iNL_x, data_type_x, iNL_y, data_type_y, dataset, iNL_r, case, miss.params$phi0, sim.params$sim_index)
  ifelse(!dir.exists(dir_name), dir.create(dir_name,recursive=T), F)
  
  diag_dir_name = sprintf("%s/Diagnostics", dir_name)
  ifelse(!dir.exists(diag_dir_name), dir.create(diag_dir_name,recursive=T), F)
  
  if(grepl("x",case)){
    # plot missing X vals (1 col)
    for(i in 1:length(miss_cols)){
      png(sprintf("%s/%s_col%d_X.png",diag_dir_name, mechanism, miss_cols[i]))
      boxplot(X[mask_x[,miss_cols[i]]==0,miss_cols[i]],
              X[mask_x[,miss_cols[i]]==1,miss_cols[i]],
              X[mask_x[,miss_cols[i]]==0,ref_cols[i]], 
              X[mask_x[,miss_cols[i]]==1,ref_cols[i]], 
              names =c(sprintf("Missing, col%d",miss_cols[i]), sprintf("Observed, col%d",miss_cols[i]),
                       sprintf("Missing, col%d",ref_cols[i]), sprintf("Observed, col%d",ref_cols[i])),
              main=sprintf("Values of miss_col%d and corr ref_col%d",miss_cols[i],ref_cols[i]),
              sub="Stratified by mask in miss_col",outline=F)
      dev.off()
    }
  }
  if(grepl("y",case)){
    png(sprintf("%s/%s_Y.png",diag_dir_name, mechanism))
    # plot missing Y vals
    boxplot(Y[mask_y==0,],
            Y[mask_y==1,],
            X[mask_y==0,ref_cols[1]], 
            X[mask_y==1,ref_cols[1]], 
            names =c("Masked, Y", "Unmasked, Y", "Masked, ref_col", "Unmasked, ref_col"),
            main="Values of Y and 1st ref_col (in X)", sub="Stratified by mask in Y",outline=F)
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
  
  save(list=c("X","Y","mask_x","mask_y","g"), file = sprintf("%s/data_%s_%d_%d.RData", dir_name, mechanism, miss.params$miss_pct_features, pi*100))
  save(list=c("sim.params","miss.params","sim.data","sim.mask"), file = sprintf("%s/params_%s_%d_%d.RData", dir_name, mechanism, miss.params$miss_pct_features, pi*100))
  print("X:")
  print(head(X,n=20))
  print("Y:")
  print(head(Y,n=20))
  print("mask_x")
  print(head(mask_x,n=20))
  print("mask_y")
  print(head(mask_y,n=20))
  return(list(X=X,Y=Y,mask_x=mask_x,mask_y=mask_y,g=g,sim.params=sim.params,miss.params=miss.params,sim.data=sim.data,sim.mask=sim.mask))
}
