x = [
    1  1  0  0  0  0  0  0  0  0  0  0
    0  1  1  0  0  0  0  0  0  0  0  0
    0  0  1  1  0  0  0  0  0  0  0  0
    0  0  0  0  1  1  0  0  0  0  0  0 
    0  0  0  0  0  0  1  0  0  0  1  0
    0  0  0  0  0  0  0  0  0  0  1  1
    1  0  0  0  0  0  1  0  0  0  0  0
    0  1  0  0  0  0  0  0  0  0  1  0
    0  0  1  0  0  0  0  0  0  0  0  1
    0  0  0  0  0  1  0  1  0  0  0  0
    0  0  0  0  0  0  1  0  1  0  0  0
]

# Included equations
a = Bool.(reduce(vcat,reduce.(hcat,Iterators.product(fill(0:1,size(x,1))...))))[2:end,:]

# Get number of equations in each
N = sum(a,dims=2)[:]

# Get number of unique variables in each
M = [sum(min.(1,sum(x[b,:],dims=1))) for b in eachrow(a)]

# What gives us N ≥ M
I = N .≥ M
A = a[(eachindex(N))[I][findmin(N[I])[2]],:]

# Which variables?
findall(sum(x[A,:],dims=1) .> 0)