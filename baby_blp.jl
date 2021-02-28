## IO Pset1 BLP
## Lei Ma, Feb 2021
cd()
cd("/Dropbox/Github/baby_BLP")
using CSV
using DataFrames
using GLM
using Statistics
using LinearAlgebra
using Distributions
using NLopt
otc = CSV.read("otc.csv", DataFrame)

###############################################################################
####  BLP fixed-point algorithm, inverting mkt shares to get mean utility  ####
###############################################################################

ns = 500;
nmkt = maximum(otc.mkt);
mkt = unique(otc.mkt);
nprod = maximum(otc.product);

vi = quantile.(Normal(), collect(range(0.5/ns, step = 1/ns, length = ns)));
sigma = 1;

function calc_mkt_share_t(delta_t, sigma_t, x_t, vi_t)
    # Dimension: delta_t 11*1, simga_t 1*1, x_t 11*1
    delta_t = delta_t .* ones(nprod, ns)
    mu_t = x_t*sigma_t*vi_t'
    numerator = exp.(delta_t .+ mu_t)
    denominator = ones(nprod, ns) .+ sum(numerator, dims = 1)
    mkt_share_t = mean(numerator./denominator, dims = 2)
end

function contraction_t(d0, sigma_t, x_t, vi_t, mkt_t, tol = 1e-5, maxiter = 1e5)
    obs_mkt_share_t = mkt_t.mkt_share
    d_old = d0
    normdiff = Inf
    iter = 0
    while normdiff > tol && iter <= maxiter
        model_mkt_share_t = calc_mkt_share_t(d_old, sigma_t, x_t, vi_t)
        d_new = d_old .+ log.(obs_mkt_share_t) .- log.(model_mkt_share_t)
        normdiff = maximum(norm.(d_new .- d_old))
        d_old = d_new
        iter += 1
    end
    return d_old
end

function calc_delta(sigma)
    delta_fp = zeros(nprod, nmkt);
    for t in mkt
        mkt_t = otc[otc.mkt .== t, :];
        x_t = ones(nprod, 1);
        delta_t = zeros(nprod, 1);
        sigma_t = sigma;
        vi_t = vi;
        delta_fp[:, t] = contraction_t(delta_t, sigma_t, x_t, vi_t, mkt_t);
    end
    return vec(delta_fp);
end

@time delta_fp = calc_delta(sigma);
mean(delta_fp)
std(delta_fp)

################################################################
#### Estimate beta and sigma using GMM (cost as instrument) ####
################################################################
X = hcat(ones(nprod*nmkt, 1),
         otc.price, otc.promotion,
         otc.product_2, otc.product_3, otc.product_4, otc.product_5,
         otc.product_6, otc.product_7, otc.product_8, otc.product_9,
         otc.product_10, otc.product_11);
z = hcat(X, otc.cost);
Phi = z'*z/1056;
inv_Phi = inv(Phi);

function GMMObjFunc(theta2::Vector, grad::Vector)
    sigma = theta2[1]
    delta = calc_delta(sigma)
    theta1 = inv(X'*z*inv_Phi*z'*X)*X'*z*inv_Phi*z'*delta
    error = delta - X*theta1
    obj = error'*z*inv_Phi*z'*error
    return obj
end

opt = Opt(:LN_COBYLA, 1)
opt.xtol_rel = 1e-4
opt.lower_bounds = [0.00001]
opt.min_objective = GMMObjFunc
@time (minf,minx,ret) = optimize(opt, [1])

@show sigma = minx[1]
delta = calc_delta(sigma[1]);
theta1 = inv(X'*z*inv_Phi*z'*X)*X'*z*inv_Phi*z'*delta
