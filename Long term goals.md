# Goals of the project

- Implement 3 disease dynamics, with 3 different self-protecting behaviors SPB:

SIR with heterogeneous susceptibility
SIR with heterogeneous vaccination probabilities
SIR with heterogeneous testing rate (increase the recovery rate)

- The population will be divided in subpopulaitons according to a beta distribution dictated by parameters A, and B.
    - each subpopulation will have a linearly increasing value of SPB from SPB_low to SPB_high

- There is a possibility for a heterogeneous mixing pattern domniated by a parameter h (homophily)
- if h is zero the mixing patter is homogenous, no homophily, and there is no need of using a contact matrix
- if h is bigger than zero each population will have a preferential attachment to populations "closer" to it in terms of behavior (homophily)
- if h is smaller than zero each population will have a preferential attachment to populations "further away" to it in terms of behavior (heterophily)


- I want to be able to swipe across several thousands combination of parameters. the combination of parameters I will explore are: 
    - (SPB_high, A = B)
    - (A, B)
    - (h, SPB_high)
    - (h, A = B)

- I want the code to be FAST. So use as much JAX as possible.

- Since JAX is not very compatible with classes, do not use classes structures