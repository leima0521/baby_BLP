## A gentle introduction to BLP

This repo includes the BLP estimation for the second-year IO class taught by [Prof. Marc Rysman](http://sites.bu.edu/mrysman/teaching/). All errors are my own.

### Setup

The estimation uses [OTC_data.xlsx](https://github.com/leima0521/baby_BLP/blob/master/OTC_data.xlsx), which includes `price`, `sales`, `promotion`, `count`, `promotion`, and `cost` for each `product` in each `week` at each `store`. There are 11 products, 2 stores, and 48 weeks. Markets are defined by store-week pairs, so there are 96 markets in total. Market size is proxied by the variable `count`. 

[make_plots.R](https://github.com/leima0521/baby_BLP/blob/master/make_plots.R) graphs the time series of sales for each brand and prices for each product. It also estimates a simple logit model without random coefficients.

### Baby BLP

[baby_blp.jl](https://github.com/leima0521/baby_BLP/blob/master/baby_blp.jl) estimates a simple BLP model on the demand side using [Julia v1.5.3](https://julialang.org/). A detailed description of the procedure can be found in [baby_blp.pdf](https://github.com/leima0521/baby_BLP/blob/master/baby_blp.pdf). There are two main parts of the code:

- The first part solves the fixed point problem to compute the mean utility $\delta$. Utility is assumed to be a linear function of `price`, `promotion`, and product fixed effects. The random coefficient term is on the constant term. I assume the variance of the random coefficient is 1 when demonstrating the fixed point algorithm.

- The second part estimates the parameters using GMM. Price is assumed to be exogenous. The instrument vector $z$ includes `price`, `promotion`, product fixed effects, and `cost`. Instead of computing the efficient weighting matrix, I just use $z'z/n$. 

