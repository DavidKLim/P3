
N=100000; P=2; pi=0.5; sim_index=1; seed=9; mechanism="MNAR"; miss_cols=1; ref_cols=2

save_toy_data = function(case=c("x","y","xy"), family="Gaussian", data_types = rep("real", P),
                         N=100000, P=2, pi=0.5, sim_index=1, seed=9, mechanism="MNAR", miss_cols = 1, ref_cols = 2){
  save.dir = sprintf("%s/toy%s_data/%s",family,case,mechanism)
  ifelse(dir.exists(save.dir),F,dir.create(save.dir,recursive=T))
  
  C=3   # 3 classes as default for all categorical X (data_types=="cat"), and for Y if family=="Multinomial"
  
  # Simulate X
  X = matrix(nrow=N, ncol=P)
  set.seed(seed)
  for(p in 1:P){
    if(data_types[p]=="real"){
      X[, p] = rnorm(N,mean=4,sd=1)
    } else if(data_types[p]=="cat"){
      #### commented out for now; just deal with Gaussian covariates for now...
      # X[, p] = apply(rmultinom(N, 1, rep(1/C,C)), 2, which.max)
      ### for categorical data, we have to create dummy variables for categories.... (in simulating Y)
    } else if(data_types[p]=="count"){
      X[, p] = rpois(N, lambda=8)
    }
  }
  
  
  # family="Gaussian" --> Gaussian data for Y
  if(family=="Gaussian"){
    # Simulate y from X --> y = Xb + e
    beta0s = 0
    betas = c(-1,1)   # effect sizes -1 and 1 test case. Must be of length P
    # e = rnorm(N,0,1)
    # Y = beta0s + X %*% betas + e
    Y = beta0s + X %*% betas
  } else if(family=="Multinomial"){
    # beta0s = rnorm(C, 0, 1)
    # betas = cbind(rnorm(C, -1, 1), rnorm(C, 1, 1))   # C x P matrix: each covariates' effects on each class
    beta0s = 0
    betas = cbind(-c(1:C),c(1:C))
    prs = exp(matrix(beta0s,nrow=N,ncol=C,byrow=T) + X %*% t(betas))
    prs = prs/rowSums(prs)
    Y = apply(prs, 1, sample, x=c(1:C), size=1, replace=F)
  } else if(family=="Poisson"){
    beta0s = 8
    betas = c(-1,1)   # effect sizes -1 and 1 test case. Must be of length P
    Y = round(exp(beta0s + X %*% betas),0)   # log(Y) = eta. Round Y to integer (to simulate count)
  }
  Y = matrix(Y,ncol=1)
  hist(Y)
  #y2 = rnorm(N*P, X%*%betas, 1)  # distrib of y and y2 should be equivalent
  #hist(y2)
  
  
  ## Example multinomial data sim
  # # covariate matrix
  # mX = matrix(rnorm(1000), 200, 5)
  # # coefficients for each choice
  # vCoef1 = rep(0, 5)
  # vCoef2 = rnorm(5)
  # vCoef3 = rnorm(5)
  # # vector of probabilities
  # vProb = cbind(exp(mX%*%vCoef1), exp(mX%*%vCoef2), exp(mX%*%vCoef3))
  # # multinomial draws
  # mChoices = t(apply(vProb, 1, rmultinom, n = 1, size = 1))
  # dfM = cbind.data.frame(y = apply(mChoices, 1, function(x) which(x==1)), mX)
  
  # Simulate R from x and/or y
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
      alph <- sapply(pi, function(y) find_int(y,phi))
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
  
  # Without missingness: coefficients look good
  if(family=="Gaussian"){
    fit = glm(Y~X)
  }else if(family=="Multinomial"){
    # fit=glm(as.factor(Y) ~ X, family = binomial(link="logit"))
    library(nnet)
    fit=multinom(Y~X)
  }else if(family=="Poisson"){
    fit=glm(Y ~ X, family = poisson(link="log"))
  }
  
  print(summary(fit))
  
  # save diagnostics, simulate missingness
  save.dir2 = sprintf("%s/Diagnostics",save.dir)
  ifelse(dir.exists(save.dir2),F,dir.create(save.dir2,recursive=T))
  
  overlap_hists=function(x1,x2,x3=NULL,lab1="Truth",lab2="Imputed",lab3="...",
                         title="MNAR Missing Values, Truth vs Imputed, Missing column",
                         save.dir, file.name){
    library(ggplot2)
    x1=data.frame(value=x1); x1$status=lab1
    x2=data.frame(value=x2); x2$status=lab2
    if(!is.null(x3)){x3=data.frame(value=x3); x3$status=lab3; df=rbind(x1,x2,x3)
    }else{df = rbind(x1,x2)}
    p = ggplot(df,aes(value,fill=status)) + geom_density(alpha=0.2) + ggtitle(title)
    ggsave(filename=sprintf("%s/%s.png",save.dir,file.name),plot=p)
    print(p)
  }
  
  Rx=matrix(1,nrow=nrow(X),ncol=ncol(X)); pRx=matrix(1,nrow=nrow(X),ncol=ncol(X))
  Ry=matrix(1,nrow=nrow(Y),ncol=ncol(Y)); pRy=matrix(1,nrow=nrow(Y),ncol=ncol(Y))
  if(case %in% c("x","xy")){
    # if MAR: ref cols and miss cols paired, each ref col is covariate for each miss col
    # if MNAR: itself is covariate for each miss col
    fit_missing = simulate_missing(data = X, miss_cols = miss_cols, ref_cols = ref_cols, pi = pi,
                                   phis = phis, phi_z=NULL, classes=NULL, scheme="UV",
                                   mechanism=mechanism, sim_index=sim_index, fmodel="S", Z=NULL)
    
    Rx = fit_missing$Missing
    pRx = fit_missing$probs
    overlap_hists(x1=X[Rx[,miss_cols[1]]==0, miss_cols[1]], lab1="Missing",
                  x2=X[Rx[,miss_cols[1]]==1, miss_cols[1]], lab2="Observed",
                  title=sprintf("%s: Values of col%d of X (miss), wrt missingness of column %d (miss)",
                                mechanism, miss_cols[1], miss_cols[1]),
                  save.dir=save.dir2, file.name="X MNAR check")
    overlap_hists(x1=X[Rx[,miss_cols[1]]==0, ref_cols[1]], lab1="Missing",
                  x2=X[Rx[,miss_cols[1]]==1, ref_cols[1]], lab2="Observed",
                  title=sprintf("%s: Values of col%d of X (obs), wrt missingness of column %d (miss)",
                                mechanism, ref_cols[1], miss_cols[1]),
                  save.dir=save.dir2, file.name="X MAR check")
  }
  # sim y missing here: y missing can be dep on y and x
  if(case %in% c("y","xy")){
    XY = data.frame(cbind(X[,ref_cols[1]],Y))   # only the fully observed X feature and missingness imposed y feature input (as covariates in MAR and MNAR, respectively)

    # if MAR: first ref_cols will be used as covariate for y's missingness
    # if MNAR: y itself will be used for y's missingness
    
    fit_missing = simulate_missing(data = XY, miss_cols = 2, ref_cols = 1,
                                   pi=pi, phis=phis, phi_z=NULL, classes=NULL, scheme="UV",
                                   mechanism=mechanism, sim_index=sim_index, fmodel="S", Z=NULL)
    Ry = matrix(fit_missing$Missing[,2],ncol=1)
    pRy = matrix(fit_missing$probs[,2],ncol=1)
    
    overlap_hists(x1=Y[Ry[,1]==0, 1], lab1="Missing",
                  x2=Y[Ry[,1]==1, 1], lab2="Observed",
                  title=sprintf("%s: Values of Y, wrt missingness of Y",
                                mechanism),
                  save.dir=save.dir2, file.name="Y MNAR check")
    overlap_hists(x1=X[Ry[,1]==0, ref_cols[1]], lab1="Missing",
                  x2=X[Ry[,1]==1, ref_cols[1]], lab2="Observed",
                  title=sprintf("%s: Values of col%d of X (obs), wrt missingness of Y",
                                mechanism, ref_cols[1]),
                  save.dir=save.dir2, file.name="Y MAR check")
  }
  
  ## Tests
  if(case=="x"){
    X0 = X; X0[Rx==0]=0
    Y0 = Y
  }else if(case=="y"){
    Y0 = Y; Y0[Ry==0]=0
    X0 = X
  }else if(case=="xy"){
    X0 = X; X0[Rx==0]=0
    Y0 = Y; Y0[Ry==0]=0
  }
  
  # Without missingness: coefficients look good
  if(family=="Gaussian"){
    fit0 = glm(Y0~X0)
  }else if(family=="Multinomial"){
    # fit=glm(as.factor(Y) ~ X, family = binomial(link="logit"))
    library(nnet)
    fit0=multinom(Y0~X0)
  }else if(family=="Poisson"){
    fit0=glm(Y0 ~ X0, family = poisson(link="log"))
  }
  
  print(summary(fit0))
  
  
  ## overlap hists with y?
  
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
  Rxs = split(data.frame(Rx), g)
  pRxs = split(data.frame(pRx),g)
  Rys = split(data.frame(Ry), g)
  pRys = split(data.frame(pRy),g)
  
  write.csv(Xs$train,file=sprintf("%s/trainX.csv",save.dir),row.names = F)
  write.csv(Ys$train,file=sprintf("%s/trainY.csv",save.dir),row.names = F)
  write.csv(Rxs$train,file=sprintf("%s/trainRx.csv",save.dir),row.names = F)
  write.csv(pRxs$train,file=sprintf("%s/trainpRx.csv",save.dir),row.names = F)
  write.csv(Rys$train,file=sprintf("%s/trainRy.csv",save.dir),row.names = F)
  write.csv(pRys$train,file=sprintf("%s/trainpRy.csv",save.dir),row.names = F)
  
  write.csv(Xs$valid,file=sprintf("%s/validX.csv",save.dir),row.names = F)
  write.csv(Ys$valid,file=sprintf("%s/validY.csv",save.dir),row.names = F)
  write.csv(Rxs$valid,file=sprintf("%s/validRx.csv",save.dir),row.names = F)
  write.csv(pRxs$valid,file=sprintf("%s/validpRx.csv",save.dir),row.names = F)
  write.csv(Rys$valid,file=sprintf("%s/validRy.csv",save.dir),row.names = F)
  write.csv(pRys$valid,file=sprintf("%s/validpRy.csv",save.dir),row.names = F)
  
  write.csv(Xs$test,file=sprintf("%s/testX.csv",save.dir),row.names = F)
  write.csv(Ys$test,file=sprintf("%s/testY.csv",save.dir),row.names = F)
  write.csv(Rxs$test,file=sprintf("%s/testRx.csv",save.dir),row.names = F)
  write.csv(pRxs$test,file=sprintf("%s/testpRx.csv",save.dir),row.names = F)
  write.csv(Rys$test,file=sprintf("%s/testRy.csv",save.dir),row.names = F)
  write.csv(pRys$test,file=sprintf("%s/testpRy.csv",save.dir),row.names = F)
  
  params = list(phis=phis,
                beta0s=beta0s,
                betas=betas,
                #e=e,
                N=N,P=P,pi=pi,sim_index=sim_index,seed=seed,mechanism=mechanism)
  save(list=c("X","Y","Rx","pRx","Ry","pRy","g","params"),file=sprintf("%s/sim_params.RData",save.dir))
}

# cases = c("x","y","xy")
cases = c("x")
mechanisms=c("MCAR","MAR","MNAR")

for(c in 1:length(cases)){
  for(m in 1:length(mechanisms)){
    # save_toy_data(family="Gaussian", mechanism=mechanisms[m], case=cases[c])
    save_toy_data(family="Multinomial", mechanism=mechanisms[m], case=cases[c])
    save_toy_data(family="Poisson", mechanism=mechanisms[m], case=cases[c])
  }
}
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