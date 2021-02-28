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

###############################
#### Compute marginal cost ####
###############################

alpha = -theta1[2];
delta_j = calc_delta(sigma);

x_t = ones(nprod, 1);
mu_ij = zeros(nprod, ns, nmkt);
for t in mkt
    mu_ij[:, :, t] = x_t * sigma * vi';
end
mu_ij = reshape(permutedims(mu_ij, [1, 3, 2]), nprod*nmkt, ns);

S_ijt = zeros(nprod, ns, nmkt);
for t in mkt
    start_row = (t-1)*11 + 1;
    end_row = 11*t;
    delta_jt = delta_j[start_row:end_row, :];
    mu_ijt = mu_ij[start_row:end_row, :];
    S_ijt_numerator = exp.(delta_jt .* ones(nprod, ns) + mu_ijt);
    S_ijt_denominator = ones(nprod, ns) .+ sum(S_ijt_numerator, dims = 1);
    S_ijt[:, :, t] = S_ijt_numerator ./ S_ijt_denominator
end
S_ijt = reshape(permutedims(S_ijt, [1, 3, 2]), nprod*nmkt, ns);

diagM = zeros(nprod, nprod);
diagM[1:3, 1:3] = ones(3, 3);
diagM[4:6, 4:6] = ones(3, 3);
diagM[7:9, 7:9] = ones(3, 3);
diagM[10:11, 10:11] = ones(2, 2);
Δ = zeros(nprod*nmkt, nprod);
mc = zeros(nprod*nmkt, 1);

for t in mkt
    start_row = (t-1)*11 + 1;
    end_row = 11*t;
    s_t = S_ijt[start_row:end_row, :];
    Δ_t = Δ[start_row:end_row, :];
    mc_t = mc[start_row:end_row, :];

    for i in 1:ns
        s_it = s_t[:, i];
        ss = alpha .* s_it * s_it';
        ss_diag = -alpha .* s_it .* (1 .- s_it);
        ss[1:(nprod+1):end] = ss_diag;
        Δ_t = Δ_t + ss;
    end

    Δ_t = Δ_t ./ ns;
    Δ_t = Δ_t .* diagM;
    Δ[start_row:end_row, :] = Δ_t;

    p_t = otc[otc.mkt .== t, :].price;
    s_jt = mean(s_t, dims = 2);
    mc_t = p_t + inv(Δ_t) * s_jt;
    mc[start_row:end_row, :] = mc_t;
end

@show mean(mc)
@show std(mc)
@show cor(mc, otc.cost)[1]
