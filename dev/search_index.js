var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = QuantumControlBase","category":"page"},{"location":"#QuantumControlBase","page":"Home","title":"QuantumControlBase","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for QuantumControlBase.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [QuantumControlBase, QuantumControlBase.ConditionalThreads]","category":"page"},{"location":"#QuantumControlBase.AbstractControlObjective","page":"Home","title":"QuantumControlBase.AbstractControlObjective","text":"Base class for a single optimization objective.\n\nAll objectives must have a field initial_state and a field generator, at minimum.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumControlBase.ControlProblem","page":"Home","title":"QuantumControlBase.ControlProblem","text":"A full control problem with multiple objectives.\n\nControlProblem(\n   objectives=<list of objectives>,\n   pulse_options=<dict of controls to pulse options>,\n   tlist=<time grid>,\n   kwargs...\n)\n\nNote that the control problem can only be instantiated via keyword arguments.\n\nThe objectives are a list of AbstractControlObjective instances, each defining an initial state and a dynamical generator for the evolution of that state. Usually, the objective will also include a target state (see Objective) and possibly a weight (see WeightedObjective.\n\nThe pulse_options are a dictionary (IdDict) mapping controls that occur in the objectives to properties specific to the control method.\n\nThe tlist is the time grid on which the time evolution of the initial states of each objective should be propagated.\n\nThe remaining kwargs are keyword arguments that are passed directly to the optimal control method. These typically include e.g. the optimization functional.\n\nThe control problem is solved by finding a set of controls that simultaneously fulfill all objectives.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumControlBase.Objective","page":"Home","title":"QuantumControlBase.Objective","text":"Standard optimization objective.\n\nObjective(;\n    initial_state=<initial_state>,\n    generator=<genenerator>,\n    target_state=<target_state>\n)\n\ndescribes an optimization objective where the time evoluation of the given initial_state under the given generator aims towards target_state. The generator here is e.g. a time-dependent Hamiltonian or Liouvillian.\n\nThe most common control problems in quantum control, e.g. state-to-state transitions or quantum gate implementations can be expressed by simultaneously fulfilling multiple objectives of this type.\n\nNote that the objective can only be instantiated via keyword arguments.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumControlBase.WeightedObjective","page":"Home","title":"QuantumControlBase.WeightedObjective","text":"Standard optimization objective with a weight.\n\nWeightedObjective(;\n    initial_state=<initial_state>,\n    generator=<genenerator>,\n    target_state=<target_state>,\n    weight=<weight>\n)\n\ninitializes a control objective like Objective, but with an additional weight parameter (a float generally between 0 and 1) that weights the objective relative to other objectives that are part of the same control problem.\n\n\n\n\n\n","category":"type"},{"location":"#Base.adjoint-Tuple{Objective}","page":"Home","title":"Base.adjoint","text":"adjoint(objective)\n\nAdjoint of a control objective. The adjoint objective contains the adjoint of the dynamical generator obj.generator, and adjoints of the obj.initial_state / obj.target_state if these exist and have an adjoint.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.blackman-Tuple{Any, Any, Any}","page":"Home","title":"QuantumControlBase.blackman","text":"Blackman window shape.\n\nblackman(t, t₀, T; a=0.16)\n\ncalculates\n\nB(t t_0 T) =\n    frac12left(\n        1 - a - cosleft(2π fract - t_0T - t_0right)\n        + a cosleft(4π fract - t_0T - t_0right)\n    right)\n\nfor a scalar t, with a = 0.16.\n\nSee http://en.wikipedia.org/wiki/Window_function#Blackman_windows\n\nA Blackman shape looks nearly identical to a Gaussian with a 6-sigma interval between t₀ and T.  Unlike the Gaussian, however, it will go exactly to zero at the edges. Thus, Blackman pulses are often preferable to Gaussians.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.box-Tuple{Any, Any, Any}","page":"Home","title":"QuantumControlBase.box","text":"Box shape (Theta-function).\n\nbox(t, t₀, T)\n\nevaluates the Heaviside (Theta-) function Theta(t) = 1 for t_0 le t le T; and Theta(t) = 0 otherwise.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.discretize-Tuple{Function, Any}","page":"Home","title":"QuantumControlBase.discretize","text":"Evaluate control at every point of tlist.\n\nvalues = discretize(control, tlist; via_midpoints=true)\n\ndiscretizes the given control to a Vector of values defined on the points of tlist.\n\nIf control is a function, it will will first be evaluated at the midpoint of tlist, see discretize_on_midpoints, and then the values on the midpoints are converted to values on tlist. This discretization is more stable than directly evaluationg the control function at the values of tlist, and ensures that repeated round-trips between discretize and discretize_on_midpoints can be done safely, see the note in the documentation of discretize_on_midpoints.\n\nThe latter can still be achieved by passing via_midpoints=false. While such a direct discretization is suitable e.g. for plotting, but it is unsuitable for round-trips between discretize and discretize_on_midpoints  (constant controls on tlist may result in a zig-zag on the intervals of tlist).\n\nIf control is a vector, it will be returned un-modified if it is of the same length as tlist. Otherwise, control must have one less value than tlist, and is assumed to be defined on the midpoins of tlist. In that case, discretize acts as the inverse of discretize_on_midpoints. See discretize_on_midpoints for how control values on tlist and control values on the intervals of tlist are related.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.discretize_on_midpoints-Union{Tuple{T}, Tuple{T, Any}} where T<:Function","page":"Home","title":"QuantumControlBase.discretize_on_midpoints","text":"Evaluate control at the midpoints of tlist.\n\nvalues = discretize_on_midpoints(control, tlist)\n\ndiscretizes the given control to a Vector of values on the midpoints of tlist. Hence, the resulting values will contain one less value than tlist.\n\nIf control is a vector of values defined on tlist (i.e., of the same length as tlist), it will be converted to a vector of values on the intervals of tlist. The value for the first and last \"midpoint\" will remain the original values at the beginning and end of tlist, in order to ensure exact bounary conditions. For all other midpoints, the value for that midpoint will be calculated by \"un-averaging\".\n\nFor example, for a control and tlist of length 5, consider the following diagram:\n\ntlist index:       1   2   3   4   5\ntlist:             ⋅   ⋅   ⋅   ⋅   ⋅   input values cᵢ (i ∈ 1..5)\n                   |̂/ ̄ ̄ ̂\\ / ̂\\ / ̂ ̄ ̄\\|̂\nmidpoints:         x     x   x     x   output values pᵢ (i ∈ 1..4)\nmidpoints index:   1     2   3     4\n\nWe will have p₁=c₁ for the first value, p₄=c₅ for the last value. For all other points, the control values cᵢ = fracp_i-1 + p_i2 are the average of the values on the midpoints. This implies the \"un-averaging\" for the midpoint values pᵢ = 2 c_i - p_i-1.\n\nnote: Note\nAn arbitrary input control array may not be compatible with the above averaging formula. In this case, the conversion will be \"lossy\" (discretize will not recover the original control array; the difference should be considered a \"discretization error\"). However, any further round-trip conversions between points and intervals are bijective and preserve the boundary conditions. In this case, the discretize_on_midpoints and discretize methods are each other's inverse. This also implies that for an optimal control procedure, it is safe to modify midpoint values. Modifying the the values on the time grid directly on the other hand may accumulate discretization errors.\n\nIf control is a vector of one less length than tlist, it will be returned unchanged, under the assumption that the input is already properly discretized.\n\nIf control is a function, the function will be directly evaluated at the midpoints marked as x in the above diagram..\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.flattop-Tuple{Any}","page":"Home","title":"QuantumControlBase.flattop","text":"Flat shape (one) with a switch-on/switch-off from zero.\n\nflattop(t; t₀, T, t_rise, t_fall=t_rise, func=:blackman)\n\nevaluates a shape function that starts at 0 at t=t₀, and ramps to to 1 during the t_rise interval. The function then remains at value 1, before ramping down to 0 again during the interval t_fall before T. For t  t₀ and t  T, the shape is zero.\n\nThe default switch-on/-off shape is half of a Blackman window (see blackman).\n\nFor func=:sinsq, the switch-on/-off shape is a sine-squared curve.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.getcontrols-Tuple{Tuple}","page":"Home","title":"QuantumControlBase.getcontrols","text":"Extract a Tuple of controls.\n\ncontrols = getcontrols(generator)\n\nextracts the controls from a single dynamical generator.\n\ncontrols = getcontrols(objectives)\n\nextracts the controls from a list of objectives (i.e., from each objective's generator)\n\nIn either case, controls that occur multiple times, either in a single generator, or throughout the different objectives, will occur only once in the result.\n\nBy default, assumes that any generator is a nested Tuple, e.g. (H0, (H1, ϵ1), (H2, ϵ2), ...) and extracts (ϵ1, ϵ2)\n\nEach control must be a valid argument for discretize.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.initobjpropwrk-Tuple{Objective, Any, Val}","page":"Home","title":"QuantumControlBase.initobjpropwrk","text":"wrk = initobjpropwrk(obj, tlist, method; kwargs...)\n\ninitializes a workspace for the propagation of a control Objective.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.liouvillian","page":"Home","title":"QuantumControlBase.liouvillian","text":"Construct a Liouvillian super-operator.\n\nℒ = liouvillian(Ĥ, c_ops=(); convention=:LvN)\n\ncalculates the sparse Liouvillian super-operator ℒ from the Hamiltonian Ĥ and a list c_ops of Lindblad operators.\n\nWith convention=:LvN, applying the resulting ℒ to a vectorized density matrix ρ⃗ calculates fracddt vecrho(t) = ℒ vecrho(t) equivalent to the Liouville-von-Neumann equation for the density matrix ρ,\n\nfracddt ρ(t)\n= -i H ρ(t) + sum_kleft(\n    A_k ρ A_k^dagger\n    - frac12 A_k^dagger A_k ρ\n    - frac12 ρ A_k^dagger A_k\n  right)\n\nwhere the Lindblad operators A_k are the elements of c_ops.\n\nThe Hamiltonian H may be time-dependent, using a nested-tuple format by default, e.g., (Ĥ₀, (H₁, ϵ₁), (H₂, ϵ₂)), where ϵ₁ and ϵ₂ are functions of time. In this case, the resulting ℒ will also be in nested tuple format, ℒ = (ℒ₀, (ℒ₁, ϵ₁), (ℒ₂, ϵ₂)), where the initial element contains the superoperator ℒ₀ for the static component of the Liouvillian, i.e., the commutator with the drift Hamiltonian Ĥ₀, plus the dissipator (sum over k), as a sparse matrix. Time-dependent Lindblad operators are not supported. The remaining elements are tuples (ℒ₁, ϵ₁) and (ℒ₂, ϵ₂) corresponding to the commutators with the two control Hamiltonians, where ℒ₁ and ℒ₂ again are sparse matrices.\n\nIf H is not time-dependent, the resulting ℒ will be a single-element tuple containing the Liouvillian as a sparse matrix, ℒ = (ℒ₀, ).\n\nWith convention=:TDSE, the Liouvillian will be constructed for the equation of motion -i hbar fracddt vecrho(t) = ℒ vecrho(t) to match exactly the form of the time-dependent Schrödinger equation. While this notation is not standard in the literature of open quantum systems, it has the benefit that the resulting ℒ can be used in a numerical propagator for a (non-Hermitian) Schrödinger equation without any change. Thus, for numerical applications, convention=:TDSE is generally preferred. The returned ℒ between the two conventions differs only by a factor of i, since we generally assume hbar=1.\n\nThe convention keyword argument is mandatory, to force a conscious choice.\n\nSee Goerz et. al. \"Optimal control theory for a unitary operation under dissipative evolution\", arXiv 1312.0111v2, Appendix B.2 for the explicit construction of the Liouvillian superoperator as a sparse matrix.\n\n\n\n\n\n","category":"function"},{"location":"#QuantumControlBase.setcontrolvals!-Union{Tuple{D}, Tuple{Any, Tuple, D}} where D<:AbstractDict","page":"Home","title":"QuantumControlBase.setcontrolvals!","text":"In-place version of setcontrolvals.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.setcontrolvals-Union{Tuple{D}, Tuple{Tuple, D}} where D<:AbstractDict","page":"Home","title":"QuantumControlBase.setcontrolvals","text":"Construct G by plugging values into a general generator.\n\nG = setcontrolvals(generator, vals_dict)\nsetcontrolvals!(G, generator, vals_dict)\n\nevaluates the specific dynamical generator G by plugging in values into the general generator according to vals_dict.\n\nThe vals_dict is a dictionary (IdDict) mapping controls as returned by getcontrols(generator) to values.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumPropagators.propagate-Tuple{Objective, Any}","page":"Home","title":"QuantumPropagators.propagate","text":"Propagate the initial state of a control objective.\n\npropagate(obj, tlist; method=:auto, controls_map=IdDict(), kwargs...)\n\npropagates obj.initial_state under the dynamics described by obj.generator.\n\nThe optional dict control_map may be given to replace the controls in obj.generator (as obtained by getcontrols) with custom functions or vectors, e.g. with the controls resulting from optimization.\n\nAll other kwargs are forwarded to the underlying propagate method for obj.initial_state.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.ConditionalThreads.@threadsif-Tuple{Any, Any}","page":"Home","title":"QuantumControlBase.ConditionalThreads.@threadsif","text":"Conditionally apply multi-threading to for loops.\n\nThis is a variation on Base.Threads.@threads that adds a run-time boolean flag to enable or disable threading. It is intended for internal use in packages building on QuantumControlBase.\n\nUsage:\n\nusing QuantumControlBase.ConditionalThreads: @threadsif\n\nfunction optimize(objectives; use_threads=true)\n    @threadsif use_threads for k = 1:length(objectives)\n    # ...\n    end\nend\n\n\n\n\n\n","category":"macro"}]
}
