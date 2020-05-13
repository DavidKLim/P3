# Simulate X
# Simulate y from X
# Simulate R from X

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