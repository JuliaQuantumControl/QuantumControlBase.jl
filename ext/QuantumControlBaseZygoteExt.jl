module QuantumControlBaseZygoteExt

using QuantumControlBase: _default_chi_via
import Zygote

import QuantumControlBase: make_automatic_chi, make_automatic_grad_J_a


function make_automatic_chi(
    J_T,
    trajectories,
    ::Val{:Zygote};
    via=_default_chi_via(trajectories)
)

    # TODO: At some point, for a large system, we could benchmark if there is
    # any benefit to making χ a closure and using LinearAlgebra.axpby! to
    # overwrite it in-place if all states are mutable.

    function zygote_chi_via_states(Ψ, trajectories)
        # The kwargs swallow any `tau` keyword argument
        function _J_T(Ψ...)
            -J_T(Ψ, trajectories)
        end
        χ = Vector{eltype(Ψ)}(undef, length(Ψ))
        ∇J = Zygote.gradient(_J_T, Ψ...)
        for (k, ∇Jₖ) ∈ enumerate(∇J)
            χ[k] = 0.5 * ∇Jₖ  # ½ corrects for gradient vs Wirtinger deriv
            # axpby!(0.5, ∇Jₖ, false, χ[k])
        end
        return χ
    end

    function zygote_chi_via_tau(Ψ, trajectories; tau=nothing, τ=tau)
        if isnothing(τ)
            msg = "`chi` returned by `make_chi` with `via=:tau` requires keyword argument tau/τ"
            throw(ArgumentError(msg))
        end
        function _J_T(τ...)
            -J_T(Ψ, trajectories; tau=τ)
        end
        χ = Vector{eltype(Ψ)}(undef, length(Ψ))
        ∇J = Zygote.gradient(_J_T, τ...)
        for (k, traj) ∈ enumerate(trajectories)
            ∂J╱∂τ̄ₖ = 0.5 * ∇J[k]  # ½ corrects for gradient vs Wirtinger deriv
            χ[k] = ∂J╱∂τ̄ₖ * traj.target_state
            # axpby!(∂J╱∂τ̄ₖ, traj.target_state, false, χ[k])
        end
        return χ
    end

    if via ≡ :states
        return zygote_chi_via_states
    elseif via ≡ :tau
        Ψ_tgt = [traj.target_state for traj in trajectories]
        if any(isnothing.(Ψ_tgt))
            error("`via=:tau` requires that all trajectories define a `target_state`")
        end
        τ_tgt = ones(ComplexF64, length(trajectories))
        Ψ_undef = similar(Ψ_tgt)
        if abs(J_T(Ψ_tgt, trajectories) - J_T(Ψ_undef, trajectories; tau=τ_tgt)) > 1e-12
            msg = "`via=:tau` in `make_chi` requires that `J_T`=$(repr(J_T)) can be evaluated solely via `tau`"
            error(msg)
        end
        return zygote_chi_via_tau
    else
        msg = "`via` must be either `:states` or `:tau`, not $(repr(via))"
        throw(ArgumentError(msg))
    end

end


function make_automatic_grad_J_a(J_a, tlist, ::Val{:Zygote})
    function automatic_grad_J_a!(∇J_a, pulsevals, tlist)
        func = pulsevals -> J_a(pulsevals, tlist)
        ∇J_a_zygote = Zygote.gradient(func, pulsevals)[1]
        copyto!(∇J_a, ∇J_a_zygote)
    end
    return automatic_grad_J_a!
end

end
