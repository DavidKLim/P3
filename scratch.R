N=100000; P=2; pi=0.5; sim_index=1; seed=9; mechanism="MNAR"

save_toy_data = function(N=100000, P=2, pi=0.5, sim_index=1, seed=9, mechanism="MNAR", miss_cols = 1, ref_cols = 2){
  save.dir = sprintf("toy_data/%s",mechanism)
  
  # Simulate X
  set.seed(seed)
  X = matrix(rnorm(N*P,mean=4,sd=1), nrow=N, ncol=P)
  
  # Simulate y from X --> y = Xb + e
  beta0 = 0
  betas = c(-1,1)   # effect sizes -1 and 1 test case. Must be of length P
  e = rnorm(N,0,1)
  Y = beta0 + X %*% betas + e
  hist(Y)
  #y2 = rnorm(N*P, X%*%betas, 1)  # distrib of y and y2 should be equivalent
  #hist(y2)
  
  # Simulate R from X
  phis = c(-2,2)
  
  simulate_missing = function(data,miss_cols,ref_cols,pi,
                              phis,phi_z,classes,
                              scheme,mechanism,sim_index=1,fmodel="S", Z=NULL){
    #sim_index=1 unless otherwise specified
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
        if(fmodel=="S"){
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
        } else if(fmodel=="PM"){
          x <- Z
          phi = phi_z # phis should be length Z
        }
      }
      alph <- sapply(pi, function(y)find_int(y,phi))
      mod = alph + x%*%phi
      
      # Pr(Missing_i = 1)
      probs = inv_logit(mod)
      prob_Missing[,miss_cols[j]] = probs
      
      # set seed for each column for reproducibility, but still different across columns
      set.seed(j+(sim_index-1)*length(miss_cols))
      Missing[,miss_cols[j]] = rbinom(n,1,probs)
      
      params[[j]]=list(phi0=alph, phi=phi, miss=miss_cols[j], ref=ref_cols[j], scheme=scheme)
    }
    
    return(list(Missing=Missing,
                probs=prob_Missing,params=params))
  }
  
  fit_missing = simulate_missing(data = X, miss_cols = miss_cols, ref_cols = ref_cols, pi = pi,
                                 phis = phis, phi_z=NULL, classes=NULL, scheme="UV",
                                 mechanism=mechanism, sim_index=sim_index, fmodel="S", Z=NULL)
  
  R = fit_missing$Missing
  pR = fit_missing$probs
  
  ## Tests
  fit = glm(Y~X)
  summary(fit)X0 = X; X0[R==0]=0
  fit0 = glm(Y~X0)
  summary(fit0)
  
  overlap_hists=function(x1,x2,x3=NULL,lab1="Truth",lab2="Imputed",lab3="...",
                         title="MNAR Missing Values, Truth vs Imputed, Missing column"){
    library(ggplot2)
    x1=data.frame(value=x1); x1$status=lab1
    x2=data.frame(value=x2); x2$status=lab2
    if(!is.null(x3)){x3=data.frame(value=x3); x3$status=lab3; df=rbind(x1,x2,x3)
    }else{df = rbind(x1,x2)}
    p = ggplot(df,aes(value,fill=status)) + geom_density(alpha=0.2) + ggtitle(title)
    print(p)
  }

  overlap_hists(x1=X[R[,miss_cols[1]]==0, miss_cols[1]], lab1="Missing",
                x2=X[R[,miss_cols[1]]==1, miss_cols[1]], lab2="Observed",
                title=sprintf("%s: histogram of missing vs observed X values of column %d, wrt missingness of column %d",
                              mechanism, miss_cols[1], miss_cols[1]))
  overlap_hists(x1=X[R[,miss_cols[1]]==0, ref_cols[1]], lab1="Missing",
                x2=X[R[,miss_cols[1]]==1, ref_cols[1]], lab2="Observed",
                title=sprintf("%s histogram of missing vs observed X values of column %d, wrt missingness of column %d",
                              mechanism, ref_cols[1], miss_cols[1]))
  
  ## Save
  ratios=c(train = .6, test = .2, valid = .2)
  set.seed(333)
  g = sample(cut(
    seq(nrow(X)), 
    nrow(X)*cumsum(c(0,ratios)),
    labels = names(ratios)
  ))
  Xs = split(data.frame(X), g)        # split by $train, $test, and $valid
  Ys = split(data.frame(Y), g)        # split by $train, $test, and $valid
  Rs = split(data.frame(R), g)
  pRs = split(data.frame(pR),g)
  
  write.csv(Xs$train,file=sprintf("%s/trainX.csv",save.dir),row.names = F)
  write.csv(Ys$train,file=sprintf("%s/trainY.csv",save.dir),row.names = F)
  write.csv(Rs$train,file=sprintf("%s/trainR.csv",save.dir),row.names = F)
  write.csv(pRs$train,file=sprintf("%s/trainpR.csv",save.dir),row.names = F)
  write.csv(Xs$valid,file=sprintf("%s/validX.csv",save.dir),row.names = F)
  write.csv(Ys$valid,file=sprintf("%s/validY.csv",save.dir),row.names = F)
  write.csv(Rs$valid,file=sprintf("%s/validR.csv",save.dir),row.names = F)
  write.csv(pRs$valid,file=sprintf("%s/validpR.csv",save.dir),row.names = F)
  write.csv(Xs$test,file=sprintf("%s/testX.csv",save.dir),row.names = F)
  write.csv(Ys$test,file=sprintf("%s/testY.csv",save.dir),row.names = F)
  write.csv(Rs$test,file=sprintf("%s/testR.csv",save.dir),row.names = F)
  write.csv(pRs$test,file=sprintf("%s/testpR.csv",save.dir),row.names = F)
  
  params = list(phis=phis,
                beta0=beta0,
                betas=betas,
                e=e,
                N=N,P=P,pi=pi,sim_index=sim_index,seed=seed,mechanism=mechanism)
  save(list=c("X","Y","R","pR","g","params"),file=sprintf("%s/sim_params.RData",save.dir))
}

save_toy_data(mechanism="MCAR"); save_toy_data(mechanism="MAR"); save_toy_data(mechanism="MNAR")
## High level overview of architecture:
# NN1: Input Xo and Xm=0, Output: Xo and Xm'
# NN2: Input Xo and Xm', Output: Y
# NN3: Input Xo and Xm', Output: R
## ELBO: No KL(q(Z)|p(Z)) term.
## Need to derive new ELBO? need to include y here
# \log p(x,r) = \log \int p(xo, xm, r) d(xm) = \log \int{ p(xo, xm, r) q(xm)/q(xm) } = \log E_q(xm)[p(xo,xm,r)/q(xm)]
# >= E_q(xm) \log {p(xo,xm,r)/q(xm)}
## log p(y,r|x ) = 

## Perhaps easier to split by case:
# Case 1: Missing response y (bios 773, slide 279, take out random effects Z)
# Case 2: Missing covariate x (bios 773, slide 357, take out random effects Z)
# Case 3: Missing both (maybe?) (bios 773, slide 357, take out random effects Z)