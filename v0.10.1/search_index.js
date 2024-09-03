var documenterSearchIndex = {"docs":
[{"location":"#QuantumControlBase","page":"Home","title":"QuantumControlBase","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The QuantumControlBase package provides methods the are useful to multiple packages within the JuliaQuantumControl organization.","category":"page"},{"location":"","page":"Home","title":"Home","text":"note: Note\nAll user-facing methods defined here are exposed in the main QuantumControl package, so please see its documentation for information on the usage of these methods in a larger context.","category":"page"},{"location":"","page":"Home","title":"Home","text":"gdeftgttexttgt gdeftroperatornametr gdefReoperatornameRe gdefImoperatornameIm","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"index.md\"]","category":"page"},{"location":"#Reference","page":"Home","title":"Reference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [QuantumControlBase]","category":"page"},{"location":"#QuantumControlBase.ControlProblem","page":"Home","title":"QuantumControlBase.ControlProblem","text":"A full control problem with multiple trajectories.\n\nControlProblem(\n   trajectories,\n   tlist;\n   kwargs...\n)\n\nThe trajectories are a list of Trajectory instances, each defining an initial state and a dynamical generator for the evolution of that state. Usually, the trajectory will also include a target state (see Trajectory) and possibly a weight. The trajectories may also be given together with tlist as a mandatory keyword argument.\n\nThe tlist is the time grid on which the time evolution of the initial states of each trajectory should be propagated. It may also be given as a (mandatory) keyword argument.\n\nThe remaining kwargs are keyword arguments that are passed directly to the optimal control method. These typically include e.g. the optimization functional.\n\nThe control problem is solved by finding a set of controls that minimize an optimization functional over all trajectories.\n\n\n\n\n\n","category":"type"},{"location":"#QuantumControlBase.Trajectory","page":"Home","title":"QuantumControlBase.Trajectory","text":"Description of a state's time evolution.\n\nTrajectory(\n    initial_state,\n    generator;\n    target_state=nothing,\n    weight=1.0,\n    kwargs...\n)\n\ndescribes the time evolution of the initial_state under a time-dependent dynamical generator (e.g., a Hamiltonian or Liouvillian).\n\nTrajectories are central to quantum control problems: an optimization functional depends on the result of propagating one or more trajectories. For example, when optimizing for a quantum gate, the optimization considers the trajectories of all logical basis states.\n\nIn addition to the initial_state and generator, a Trajectory may include data relevant to the propagation and to evaluating a particular optimization functional. Most functionals have the notion of a \"target state\" that the initial_state should evolve towards, which can be given as the target_state keyword argument. In some functionals, different trajectories enter with different weights [1], which can be given as a weight keyword argument. Any other keyword arguments are also available to a functional as properties of the Trajectory .\n\nA Trajectory can also be instantiated using all keyword arguments.\n\nProperties\n\nAll keyword arguments used in the instantiation are available as properties of the Trajectory. At a minimum, this includes initial_state, generator, target_state, and weight.\n\nBy convention, properties with a prop_ prefix, e.g., prop_method, will be taken into account when propagating the trajectory. See propagate_trajectory for details.\n\n\n\n\n\n","category":"type"},{"location":"#Base.adjoint-Tuple{Trajectory}","page":"Home","title":"Base.adjoint","text":"Construct the adjoint of a Trajectory.\n\nadj_trajectory = adjoint(trajectory)\n\nThe adjoint trajectory contains the adjoint of the dynamical generator traj.generator. All other fields contain a copy of the original field value.\n\nThe primary purpose of this adjoint is to facilitate the backward propagation under the adjoint generator that is central to gradient-based optimization methods such as GRAPE and Krotov's method.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.chain_callbacks-Tuple","page":"Home","title":"QuantumControlBase.chain_callbacks","text":"Combine multiple callback functions.\n\nchain_callbacks(funcs...)\n\ncombines funcs into a single Function that can be passes as callback to ControlProblem or any optimize-function.\n\nEach function in func must be a suitable callback by itself. This means that it should receive the optimization workspace object as its first positional parameter, then positional parameters specific to the optimization method, and then an arbitrary number of data parameters. It must return either nothing or a tuple of \"info\" objects (which will end up in the records field of the optimization result).\n\nWhen chaining callbacks, the funcs will be called in series, and the \"info\" objects will be accumulated into a single result tuple. The combined results from previous funcs will be given to the subsequent funcs as data parameters. This allows for the callbacks in the chain to communicate.\n\nThe chain will return the final combined result tuple, or nothing if all funcs return nothing.\n\nnote: Note\nWhen calling optimize, any callback that is a tuple will be automatically processed with chain_callbacks. Thus, chain_callbacks rarely has to be invoked manually.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.check_amplitude-Tuple{Any}","page":"Home","title":"QuantumControlBase.check_amplitude","text":"Check an amplitude in a Generator in the context of optimal control.\n\n@test check_amplitude(\n    ampl; tlist, for_gradient_optimization=true, quiet=false\n)\n\nverifies that the given ampl is a valid element in the list of amplitudes of a Generator object. This checks all the conditions of QuantumPropagators.Interfaces.check_amplitude. In addition, the following conditions must be met.\n\nIf for_gradient_optimization:\n\nThe function get_control_deriv(ampl, control) must be defined\nIf ampl does not depend on control, get_control_deriv(ampl, control) must return 0.0\nIf ampl depends on control, u = get_control_deriv(ampl, control) must return an object u so that evaluate(u, tlist, n) returns a Number. In most cases, u itself will be a Number. For more unusual amplitudes, e.g., an amplitude with a non-linear dependency on the controls, u may be another amplitude. The controls in u (as obtained by QuantumPropagators.Controls.get_controls) must be a subset of the controls in ampl.\n\nThe function returns true for a valid amplitude and false for an invalid amplitude. Unless quiet=true, it will log an error to indicate which of the conditions failed.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.check_generator-Tuple{Any}","page":"Home","title":"QuantumControlBase.check_generator","text":"Check the dynamical generator in the context of optimal control.\n\n@test check_generator(\n    generator; state, tlist,\n    for_expval=true, for_pwc=true, for_time_continuous=false,\n    for_parameterization=false, for_gradient_optimization=true,\n    atol=1e-15, quiet=false\n)\n\nverifies the given generator. This checks all the conditions of QuantumPropagators.Interfaces.check_generator. In addition, the following conditions must be met.\n\nIf for_gradient_optimization:\n\nget_control_derivs(generator, controls) must be defined and return a vector containing the result of get_control_deriv(generator, control) for every control in controls.\nget_control_deriv(generator, control) must return an object that passes the less restrictive QuantumPropagators.Interfaces.check_generator if control is in get_controls(generator). The controls in the derivative (if any) must be a subset of the controls in generator.\nget_control_deriv(generator, control) must return nothing if control is not in get_controls(generator)\nIf generator is a Generator instance, every ampl in generator.amplitudes must pass check_amplitude(ampl; tlist).\n\nThe function returns true for a valid generator and false for an invalid generator. Unless quiet=true, it will log an error to indicate which of the conditions failed.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.get_control_deriv-Tuple{Function, Any}","page":"Home","title":"QuantumControlBase.get_control_deriv","text":"a = get_control_deriv(ampl, control)\n\nreturns the derivative a_l(t)ϵ_l(t) of the given amplitude a_l(ϵ_l(t) t) with respect to the given control ϵ_l(t). For \"trivial\" amplitudes, where a_l(t)  ϵ_l(t), the result with be either 1.0 or 0.0 (depending on whether ampl ≡ control). For non-trivial amplitudes, the result may be another amplitude that depends on the controls and potentially on time, but can be evaluated to a constant with evaluate.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.get_control_deriv-Tuple{Tuple, Any}","page":"Home","title":"QuantumControlBase.get_control_deriv","text":"Get the derivative of the generator G w.r.t. the control ϵ(t).\n\nμ  = get_control_deriv(generator, control)\n\nreturns nothing if the generator (Hamiltonian or Liouvillian) does not depend on control, or a generator\n\nμ = fracGϵ(t)\n\notherwise. For linear control terms, μ will be a static operator, e.g. an AbstractMatrix or an Operator. For non-linear controls, μ will be time-dependent, e.g. a Generator. In either case, evaluate should be used to evaluate μ into a constant operator for particular values of the controls and a particular point in time.\n\nFor constant generators, e.g. an Operator, the result is always nothing.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.get_control_derivs-Tuple{Any, Any}","page":"Home","title":"QuantumControlBase.get_control_derivs","text":"Get a vector of the derivatives of generator w.r.t. each control.\n\nget_control_derivs(generator, controls)\n\nreturn as vector containing the derivative of generator with respect to each control in controls. The elements of the vector are either nothing if generator does not depend on that particular control, or a function μ(α) that evaluates the derivative for a particular value of the control, see get_control_deriv.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.init_prop_trajectory-Tuple{Trajectory, Any}","page":"Home","title":"QuantumControlBase.init_prop_trajectory","text":"Initialize a propagator for a given Trajectory.\n\npropagator = init_prop_trajectory(\n    traj,\n    tlist;\n    initial_state=traj.initial_state,\n    kwargs...\n)\n\ninitializes a Propagator for the propagation of the initial_state under the dynamics described by traj.generator.\n\nAll keyword arguments are forwarded to QuantumPropagators.init_prop, with default values from any property of traj with a prop_ prefix. That is, the keyword arguments for the underlying QuantumPropagators.init_prop are determined as follows:\n\nFor any property of traj whose name starts with the prefix prop_, strip the prefix and use that property as a keyword argument for init_prop. For example, if traj.prop_method is defined, method=traj.prop_method will be passed to init_prop. Similarly, traj.prop_inplace would be passed as inplace=traj.prop_inplace, etc.\nAny explicitly keyword argument to init_prop_trajectory overrides the values from the properties of traj.\n\nNote that the propagation method in particular must be specified, as it is a mandatory keyword argument in QuantumPropagators.propagate). Thus, either traj must have a property prop_method of the trajectory, or method must be given as an explicit keyword argument.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.make_chi-Tuple{Any, Any}","page":"Home","title":"QuantumControlBase.make_chi","text":"Return a function that calculates χ_k = -J_TΨ_k.\n\nchi = make_chi(\n    J_T,\n    trajectories;\n    mode=:any,\n    automatic=:default,\n    via=(any(isnothing(t.target_state) for t in trajectories) ? :states : :tau),\n)\n\ncreates a function chi(Ψ, trajectories; τ) that returns a vector of states χ with χ_k = -J_TΨ_k, where Ψ_k is the k'th element of Ψ. These are the states used as the boundary condition for the backward propagation propagation in Krotov's method and GRAPE. Each χₖ is defined as a matrix calculus Wirtinger derivative,\n\nχ_k(T) = -fracJ_TΨ_k = -frac12 _Ψ_k J_Tqquad\n_Ψ_k J_T  fracJ_TReΨ_k + i fracJ_TImΨ_k\n\nThe function J_T must take a vector of states Ψ and a vector of trajectories as positional parameters. If via=:tau, it must also a vector tau as a keyword argument, see e.g. J_T_sm). that contains the overlap of the states Ψ with the target states from the trajectories\n\nThe derivative can be calculated analytically of automatically (via automatic differentiation) depending on the value of mode. For mode=:any, an analytic derivative is returned if available, with a fallback to an automatic derivative.\n\nIf mode=:analytic, return an analytically known -J_TΨ_k, e.g.,\n\nQuantumControl.Functionals.J_T_sm → QuantumControl.Functionals.chi_sm,\nQuantumControl.Functionals.J_T_re → QuantumControl.Functionals.chi_re,\nQuantumControl.Functionals.J_T_ss → QuantumControl.Functionals.chi_ss.\n\nand throw an error if no analytic derivative is known.\n\nIf mode=:automatic, return an automatic derivative (even if an analytic derivative is known). The calculation of an automatic derivative  (whether via mode=:any or mode=:automatic) requires that a suitable framework (e.g., Zygote or FiniteDifferences) has been loaded. The loaded module must be passed as automatic keyword argument. Alternatively, it can be registered as a default value for automatic by calling QuantumControl.set_default_ad_framework.\n\nWhen evaluating χ_k automatically, if via=:states is given , χ_k(T) is calculated directly as defined above from the gradient with respect to the states Ψ_k(T).\n\nIf via=:tau is given instead, the functional J_T is considered a function of overlaps τ_k = Ψ_k^tgtΨ_k(T). This requires that all trajectories define a target_state and that J_T calculates the value of the functional solely based on the values of tau passed as a keyword argument.  With only the complex conjugate τ_k = Ψ_k(T)Ψ_k^tgt having an explicit dependency on Ψ_k(T),  the chain rule in this case is\n\nχ_k(T)\n= -fracJ_TΨ_k\n= -left(\n    fracJ_Tτ_k\n    fracτ_kΨ_k\n  right)\n= - frac12 (_τ_k J_T) Ψ_k^tgt\n\nAgain, we have used the definition of the Wirtinger derivatives,\n\nbeginalign*\n    fracJ_Tτ_k\n     frac12left(\n        frac J_T Reτ_k\n        - i frac J_T Imτ_k\n    right)\n    fracJ_Tτ_k\n     frac12left(\n        frac J_T Reτ_k\n        + i frac J_T Imτ_k\n    right)\nendalign*\n\nand the definition of the Zygote gradient with respect to a complex scalar,\n\n_τ_k J_T = left(\n    frac J_T Reτ_k\n    + i frac J_T Imτ_k\nright)\n\ntip: Tip\nIn order to extend make_chi with an analytic implementation for a new J_T function, define a new method make_analytic_chi like so:QuantumControlBase.make_analytic_chi(::typeof(J_T_sm), trajectories) = chi_smwhich links make_chi for QuantumControl.Functionals.J_T_sm to QuantumControl.Functionals.chi_sm.\n\nwarning: Warning\nZygote is notorious for being buggy (silently returning incorrect gradients). Always test automatic derivatives against finite differences and/or other automatic differentiation frameworks.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.make_grad_J_a-Tuple{Any, Any}","page":"Home","title":"QuantumControlBase.make_grad_J_a","text":"Return a function to evaluate J_aϵ_ln for a pulse value running cost.\n\ngrad_J_a! = make_grad_J_a(\n    J_a,\n    tlist;\n    mode=:any,\n    automatic=:default,\n)\n\nreturns a function so that grad_J_a!(∇J_a, pulsevals, tlist) sets J_aϵ_ln as the elements of the (vectorized) ∇J_a. The function J_a must have the interface J_a(pulsevals, tlist), see, e.g., J_a_fluence.\n\nThe parameters mode and automatic are handled as in make_chi, where mode is one of :any, :analytic, :automatic, and automatic is he loaded module of an automatic differentiation framework, where :default refers to the framework set with QuantumControl.set_default_ad_framework.\n\ntip: Tip\nIn order to extend make_grad_J_a with an analytic implementation for a new J_a function, define a new method make_analytic_grad_J_a like so:make_analytic_grad_J_a(::typeof(J_a_fluence), tlist) = grad_J_a_fluence!which links make_grad_J_a for J_a_fluence to grad_J_a_fluence!.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.make_print_iters-Tuple{Module}","page":"Home","title":"QuantumControlBase.make_print_iters","text":"Construct a method-specific automatic callback for printing iter information.\n\nprint_iters = make_print_iters(Method; kwargs...)\n\nconstructs the automatic callback to be used by optimize(problem; method=Method, print_iters=true) to print information after each iteration. The keyword arguments are those used to instantiate problem and those explicitly passed to optimize.\n\nOptimization methods should implement make_print_iters(::Val{:Method}; kwargs...) where :Method is the name of the module/package implementing the method.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.optimize-Tuple{ControlProblem}","page":"Home","title":"QuantumControlBase.optimize","text":"Optimize a quantum control problem.\n\nresult = optimize(\n    problem;\n    method,  # mandatory keyword argument\n    check=true,\n    callback=nothing,\n    print_iters=true,\n    kwargs...\n)\n\noptimizes towards a solution of given problem with the given method, which should be a Module implementing the method, e.g.,\n\nusing Krotov\nresult = optimize(problem; method=Krotov)\n\nIf check is true (default), the initial_state and generator of each trajectory is checked with check_state and check_generator. Any other keyword argument temporarily overrides the corresponding keyword argument in problem. These arguments are available to the optimizer, see each optimization package's documentation for details.\n\nThe callback can be given as a function to be called after each iteration in order to analyze the progress of the optimization or to modify the state of the optimizer or the current controls. The signature of callback is method-specific, but callbacks should receive a workspace objects as the first parameter as the first argument, the iteration number as the second parameter, and then additional method-specific parameters.\n\nThe callback function may return a tuple of values, and an optimization method should store these values fore each iteration in a records field in their Result object. The callback should be called once with an iteration number of 0 before the first iteration. The callback can also be given as a tuple of vector of functions, which are automatically combined via chain_callbacks.\n\nIf print_iters is true (default), an automatic callback is created via the method-specific make_print_iters to print the progress of the optimization after each iteration. This automatic callback runs after any manually given callback.\n\nAll remaining keyword argument are method-specific. To obtain the documentation for which options a particular method uses, run, e.g.,\n\n? optimize(problem, ::Val{:Krotov})\n\nwhere :Krotov is the name of the module implementing the method. The above is also the method signature that a Module wishing to implement a control method must define.\n\nThe returned result object is specific to the optimization method.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.propagate_trajectories-Tuple{Any, Any}","page":"Home","title":"QuantumControlBase.propagate_trajectories","text":"Propagate multiple trajectories in parallel.\n\nresult = propagate_trajectories(\n    trajectories, tlist; use_threads=true, kwargs...\n)\n\nruns propagate_trajectory for every trajectory in trajectories, collects and returns a vector of results. The propagation happens in parallel if use_threads=true (default). All keyword parameters are passed to propagate_trajectory, except that if initial_state is given, it must be a vector of initial states, one for each trajectory. Likewise, to pass pre-allocated storage arrays to storage, a vector of storage arrays must be passed. A simple storage=true will still work to return a vector of storage results.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.propagate_trajectory-Tuple{Any, Any}","page":"Home","title":"QuantumControlBase.propagate_trajectory","text":"Propagate a Trajectory.\n\npropagate_trajectory(\n    traj,\n    tlist;\n    initial_state=traj.initial_state,\n    kwargs...\n)\n\npropagates initial_state under the dynamics described by traj.generator. It takes the same keyword arguments as QuantumPropagators.propagate, with default values from any property of traj with a prop_ prefix (prop_method, prop_inplace, prop_callback, …). See init_prop_trajectory for details.\n\nNote that method (a mandatory keyword argument in QuantumPropagators.propagate) must be specified, either as a property prop_method of the trajectory, or by passing a method keyword argument explicitly.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.set_atexit_save_optimization-Tuple{Any, Any}","page":"Home","title":"QuantumControlBase.set_atexit_save_optimization","text":"Register a callback to dump a running optimization to disk on unexpected exit.\n\nA long-running optimization routine may use\n\nif !isnothing(atexit_filename)\n    set_atexit_save_optimization(\n        atexit_filename, result; msg_property=:message, msg=\"Abort: ATEXIT\"\n    )\n    # ...\n    popfirst!(Base.atexit_hooks)  # remove callback\nend\n\nto register a callback that writes the given result object to the given filename in JLD2 format in the event that the program terminates unexpectedly. The idea is to avoid data loss if the user presses CTRL-C in a non-interactive program (SIGINT), or if the process receives a SIGTERM from an HPC scheduler because the process has reached its allocated runtime limit. Note that the callback cannot protect against data loss in all possible scenarios, e.g., a SIGKILL will terminate the program without giving the callback a chance to run (as will yanking the power cord).\n\nAs in the above example, the optimization routine should make set_atexit_save_optimization conditional on an atexit_filename keyword argument, which is what QuantumControl.@optimize_or_load will pass to the optimization routine. The optimization routine must remove the callback from Base.atexit_hooks when it exits normally. Note that in an interactive context, CTRL-C will throw an InterruptException, but not cause a shutdown. Optimization routines that want to prevent data loss in this situation should handle the InterruptException and return result, in addition to using set_atexit_save_optimization.\n\nIf msg_property is not nothing, the given msg string will be stored in the corresponding property of the (mutable) result object before it is written out.\n\nThe resulting JLD2 file is compatible with QuantumControl.load_optimization.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.taus!-Tuple{Vector{ComplexF64}, Any, Any}","page":"Home","title":"QuantumControlBase.taus!","text":"Overlaps of target states with propagates states, calculated in-place.\n\ntaus!(τ, Ψ, trajectories; ignore_missing_target_state=false)\n\noverwrites the complex vector τ with the results of taus(Ψ, trajectories).\n\nThrows an ArgumentError if any of trajectories have a target_state of nothing. If ignore_missing_target_state=true, values in τ instead will remain unchanged for any trajectories with a missing target state.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.taus-Tuple{Any, Any}","page":"Home","title":"QuantumControlBase.taus","text":"Overlaps of target states with propagates states\n\nτ = taus(Ψ, trajectories)\n\ncalculates a vector of values τ_k = Ψ_k^tgtΨ_k where Ψ_k^tgt is the traj.target_state of the k'th element of trajectories and Ψₖ is the k'th element of Ψ.\n\nThe definition of the τ values with Ψ_k^tgt on the left (overlap of target states with propagated states, as opposed to overlap of propagated states with target states) matches Refs. [2] and [3].\n\nThe function requires that each trajectory defines a target state. See also taus! for an in-place version that includes well-defined error handling for any trajectories whose target_state property is nothing.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumPropagators.Controls.get_controls-Tuple{ControlProblem}","page":"Home","title":"QuantumPropagators.Controls.get_controls","text":"controls = get_controls(problem)\n\nextracts the controls from problem.trajectories.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumPropagators.Controls.get_controls-Tuple{Vector{<:Trajectory}}","page":"Home","title":"QuantumPropagators.Controls.get_controls","text":"controls = get_controls(trajectories)\n\nextracts the controls from a list of trajectories (i.e., from each trajectory's generator). Controls that occur multiple times in the different trajectories will occur only once in the result.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumPropagators.Controls.get_parameters-Tuple{ControlProblem}","page":"Home","title":"QuantumPropagators.Controls.get_parameters","text":"parameters = get_parameters(problem)\n\nextracts the parameters from problem.trajectories.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumPropagators.Controls.get_parameters-Tuple{Vector{<:Trajectory}}","page":"Home","title":"QuantumPropagators.Controls.get_parameters","text":"parameters = get_parameters(trajectories)\n\ncollects and combines get parameter arrays from all the generators in trajectories. Note that this allows any custom generator type to define a custom get_parameters method to override the default of obtaining the parameters recursively from the controls inside the generator.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumPropagators.Controls.substitute-Tuple{ControlProblem, Any}","page":"Home","title":"QuantumPropagators.Controls.substitute","text":"problem = substitute(problem::ControlProblem, replacements)\n\nsubstitutes in problem.trajectories\n\n\n\n\n\n","category":"method"},{"location":"#QuantumPropagators.Controls.substitute-Tuple{Trajectory, Any}","page":"Home","title":"QuantumPropagators.Controls.substitute","text":"trajectory = substitute(trajectory::Trajectory, replacements)\ntrajectories = substitute(trajectories::Vector{<:Trajectory}, replacements)\n\nrecursively substitutes the initial_state, generator, and target_state.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumControlBase.@threadsif-Tuple{Any, Any}","page":"Home","title":"QuantumControlBase.@threadsif","text":"Conditionally apply multi-threading to for loops.\n\nThis is a variation on Base.Threads.@threads that adds a run-time boolean flag to enable or disable threading. It is intended for internal use in packages building on QuantumControlBase.\n\nUsage:\n\nusing QuantumControlBase: @threadsif\n\nfunction optimize(trajectories; use_threads=true)\n    @threadsif use_threads for k = 1:length(trajectories)\n    # ...\n    end\nend\n\n\n\n\n\n","category":"macro"},{"location":"#Bibliography","page":"Home","title":"Bibliography","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"M. H. Goerz, D. M. Reich and C. P. Koch. Optimal control theory for a unitary operation under dissipative evolution. New J. Phys. 16, 055012 (2014).\n\n\n\nJ. P. Palao and R. Kosloff. Optimal control theory for unitary transformations. Phys. Rev. A 68, 062308 (2003).\n\n\n\nM. H. Goerz, S. C. Carrasco and V. S. Malinovsky. Quantum Optimal Control via Semi-Automatic Differentiation. Quantum 6, 871 (2022).\n\n\n\n","category":"page"}]
}