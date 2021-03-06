module WeylChamber

export D_PE,
    canonical_gate,
    gate_concurrence,
    in_weyl_chamber,
    local_invariants,
    unitarity,
    weyl_chamber_coordinates,
    weyl_chamber_region


using LinearAlgebra


const ð = 1im

const Q_magic = [
    1  0  0  ð
    0  ð  1  0
    0  ð -1  0
    1  0  0 -ð
]

const SxSx = ComplexF64[
    0  0  0  1
    0  0  1  0
    0  1  0  0
    1  0  0  0
]

const SySy = ComplexF64[
     0  0  0 -1
     0  0  1  0
     0  1  0  0
    -1  0  0  0
]


const SzSz = ComplexF64[
    1  0  0  0
    0 -1  0  0
    0  0 -1  0
    0  0  0  1
]


"""Calculate the local invariants gâ, gâ, gâ for a two-qubit gate.

```julia
gâ, gâ, gâ = local_invariants(U)
```
"""
function local_invariants(U)
    Q = Q_magic
    UB = Q' * U * Q  # "to-magic"
    detU = det(UB)
    m = transpose(UB) * UB
    gââ = tr(m)^2 / 16detU
    gâ = real(gââ)
    gâ = imag(gââ)
    gâ = real((tr(m)^2 - tr(m * m)) / 4detU)
    return gâ, gâ, gâ
end


"""Calculate the Weyl chamber coordinates câ, câ, câ for a two-qubit gate.

```julia
câ, câ, câ = weyl_chamber_coordinates(U)
```

calculates the Weyl chamber coordinates using the algorithm described in Childs
et al., PRA 68, 052311 (2003).
"""
function weyl_chamber_coordinates(U)

    @assert size(U) == (4, 4)
    detU = Complex(det(U))
    UÌ = SySy * transpose(U) * SySy
    two_S = [angle(z) / Ï for z in eigvals(U * UÌ / âdetU)]

    # Check whether the argument of the eigenvalues are on correct branch. If
    # not, put them on correct branch: `angle` returns values in (-Ï,Ï] whereas
    # we need values on the branch (-Ï/2,3Ï/2]. This implies that those
    # arguments which are between -Ï and -Ï/2 need to be shifted by 2Ï
    two_S = [(Ïâ±¼ + 0.5 < -1e-10 ? Ïâ±¼ + 2 : Ïâ±¼) for Ïâ±¼ â two_S]

    p = sortperm(two_S, rev=true)  # Zygote can't handle a direct `sort`
    S = [two_S[p[1]] / 2, two_S[p[2]] / 2, two_S[p[3]] / 2, two_S[p[4]] / 2]
    n = Int(round(sum(S)))  # number of Ïâ±¼ â¤ -Ï/4
    @assert n â¥ 0
    if n > 0
        S = [j <= n ? Ïâ±¼ - 1.0 : Ïâ±¼ for (j, Ïâ±¼) in enumerate(S)]
        S = circshift(S, -n)
    end
    câ = S[1] + S[2]
    câ = S[1] + S[3]
    câ = S[2] + S[3]
    if câ < 0
        câ = 1 - câ
        câ = -câ
    end
    return câ, câ, câ

end


"""Calculate the maximum gate concurrence.

```julia
C = gate_concurrence(U)
C = gate_concurrence(câ, câ, câ)
```

calculates that maximum concurrence ``C â [0, 1]`` that the two two-qubit gate `U`,
respectively the gate described by the Weyl chamber coordinates `câ`, `câ`,
`câ` (see [`weyl_chamber_coordinates`](@ref)) can generate.

See Kraus, Cirac, Phys. Rev. A 63, 062309 (2001)
"""
function gate_concurrence(câ, câ, câ)
    if (câ + câ â¥ 0.5) && (câ - câ â¤ 0.5) && (câ + câ â¤ 0.5)
        # If we're inside the perfect-entangler polyhedron in the Weyl chamber
        # the concurrence is 1 by definition. The "regular" formula gives wrong
        # results in this case.
        C = 1.0
    else
        # Outside of the polyhedron, the Formula of Eq (21) in PRA 63, 062309
        # is valid
        return max(
            abs(sin(Ï * (câ + câ))),
            abs(sin(Ï * (câ + câ))),
            abs(sin(Ï * (câ + câ))),
            abs(sin(Ï * (câ - câ))),
            abs(sin(Ï * (câ - câ))),
            abs(sin(Ï * (câ - câ))),
        )
    end
end

gate_concurrence(U) = gate_concurrence(weyl_chamber_coordinates(U)...)



@doc raw"""Construct the canonical gate for the given Weyl chamber coordinates.

```julia
UÌ = canonical_gate(câ, câ, câ)
```

constructs the two qubit gate ``UÌ`` as

```math
UÌ = \exp\left[i\frac{Ï}{2} (c_1 ÏÌ_x ÏÌ_x + c_2 ÏÌ_y ÏÌ_y + c_3 ÏÌ_z ÏÌ_z)\right]
```

where ``ÏÌ_{x,y,z}`` are the Pauli matrices.
"""
function canonical_gate(câ, câ, câ)
    return exp(ð * Ï / 2 * (câ * SxSx + câ * SySy + câ * SzSz))
end


"""Identify which region of the Weyl chamber a given gate is located in.

```julia
region = weyl_chamber_region(U)
region = weyl_chamber_region(câ, câ, câ)
```

identifies which region of the Weyl chamber the given two-qubit gate `U`,
respectively the gate identified by the Weyl chamber coordinates `câ`, `câ`,
`câ` (see [`weyl_chamber_coordinates`](@ref)) is in, as a string. Possible
outputs are:

* `"PE"`: gate is in the polyhedron of perfect entanglers.
* `"W0"`: gate is between the identity and the perfect entanglers.
* `"W0*"`: gate is between CPHASE(2Ï) and the perfect entanglers.
* `"W1"`: gate is between SWAP and the perfect entanglers.

For invalid Weyl chamber coordinates, an empty string is returned.
"""
function weyl_chamber_region(câ, câ, câ)
    in_weyl_chamber = (
        ((0 â¤ câ < 0.5) && (0 â¤ câ â¤ câ) && (0 â¤ câ â¤ câ)) ||
        ((0.5 â¤ câ â¤ 1.0) && (0 â¤ câ â¤ (1 - câ)) && (0 â¤ câ â¤ câ))
    )
    if in_weyl_chamber
        if ((câ + câ) â¥ 0.5) && ((câ - câ) â¤ 0.5) && ((câ + câ) â¤ 0.5)
            return "PE"
        elseif (câ + câ) < 0.5
            return "W0"
        elseif (câ - câ) > 0.5
            return "W0*"
        elseif (câ + câ) > 0.5
            return "W1"
        else
            throw(ErrorException("Internal Error: ($câ, $câ, $câ) not handled"))
        end
    else
        return ""
    end
end

weyl_chamber_region(U) = weyl_chamber_region(weyl_chamber_coordinates(U)...)


"""Check whether a given gate is in (a specific region of) the Weyl chamber.

```
in_weyl_chamber(câ, câ, câ)
```

checks whether `câ`, `câ`, `câ` are valid Weyl chamber coordinates.

```
in_weyl_chamber(U; region="PE")
in_weyl_chamber(câ, câ, câ; region="PE")
```

checks whether the two-qubit gate `U`, respectively the gate described by the
Weyl chamber coordinates `câ`, `câ`, `câ` (see
[`weyl_chamber_coordinates`](@ref)) is a perfect entangler. The `region` can be
any other of the regions returned by [`weyl_chamber_region`](@ref).
"""
function in_weyl_chamber(câ, câ, câ; region="W")
    regions = ["W", "PE", "W0", "W0*", "W1"]
    if region â regions
        throw(ArgumentError("Invalid region $(repr(region)), must be one of $regions"))
    end
    found_region = weyl_chamber_region(câ, câ, câ)
    if region == "W"
        return found_region â  ""
    else
        return found_region == region
    end
end


in_weyl_chamber(U; region) = in_weyl_chamber(weyl_chamber_coordinates(U)...; region)


@doc raw"""Unitarity of a matrix.

```
pop_loss = 1 - unitarity(U)
```

measures the loss of population from the subspace described by
`U`. E.g., for a two-qubit gate, `U` is a 4Ã4 matrix. The `unitarity` is
defined as ``\Re[\tr(UÌ^â UÌ) / N]`` where ``N`` is the dimension of ``UÌ``.
"""
function unitarity(U)
    N = size(U)[1]
    return real(tr(U' * U) / N)
end


@doc raw"""Perfect-entanglers distance measure.

```julia
D = D_PE(U; unitarity_weight=0.0, absolute_square=false)
```

For a given two-qubit gate ``UÌ``, this is defined via the
[`local_invariants`](@ref) ``g_1``, ``g_2``, ``g_3`` as

```math
D = g_3 \sqrt{g_1^2 + g_2^2} - g_1
```

This describes the geometric distance of the quantum gate from the polyhedron
of perfect entanglers in the Weyl chamber.

This equation is only meaningful under the assumption that ``UÌ`` is unitary. If
the two-qubit level are a logical subspace embedded in a larger physical
Hilbert space, loss of population from the logical subspace may lead to a
non-unitary ``UÌ``. In this case, the [`unitarity`](@ref) measure can be added
to the functional by giving a `unitary_weight` â [0, 1) that specifies the
relative proportion of the ``D`` term and the unitarity term.

By specifying `absolute_square=true`, the functional is modified as ``D â
|D|Â²``, optimizing specifically for the *boundary* of the perfect entanglers
polyhedron. This accounts for the fact that ``D`` can take negative values
inside the polyhedron, as well as the `W1` region of the Weyl chamber (the one
adjacent to SWAP). This may be especially useful in a system with population
loss (`unitarity_weight` > 0), as it avoids situations where the optimization
goes deeper into the perfect entanglers while *increasing* population loss.

!!! warning

    The functional does not check which region of the Weyl chamber the quantum
    gate is in. When using this for an optimization where the guess leads to a
    point in the `W1` region of the Weyl chamber (close to SWAP), the sign of
    the functional must be flipped, or else it will optimize for SWAP.
    Alternatively, use `absolute_square=true`.

!!! tip

    The functional can be converted into the correct form for an optimization
    that uses one objective for each logical basis state by using
    `QuantumControl.Functionals.gate_functional`.
"""
function D_PE(U; unitarity_weight=0.0, absolute_square=false)
    w::Float64 = clamp(1.0 - unitarity_weight, 0.0001, 1.0)
    N = 4
    @assert size(U) == (N, N)
    gâ, gâ, gâ = local_invariants(U)
    if absolute_square
        D = abs2(gâ * sqrt(gâ^2 + gâ^2) - gâ)
    else
        D = gâ * sqrt(gâ^2 + gâ^2) - gâ
    end
    pop_loss = 1 - unitarity(U)
    return w * D + (1 - w) * pop_loss
end

end
