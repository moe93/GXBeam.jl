"""
    Material(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23, rho, 
        S1t=1.0, S1c=1.0, S2t=1.0, S2c=1.0, S3t=1.0, S3c=1.0, S12=1.0, S13=1.0, S23=1.0)

General orthotropic material properties. 
1 is along main ply axis. 2 is transverse. 3 is normal to ply.
for a fiber orientation of zero, 1 is along the beam axis.
strength properties are optional

**Arguments**
- `Ei:float`: Young's modulus along 1st, 2nd and 3rd axes.
- `Gij::float`: shear moduli
- `nuij::float`: Poisson's ratio.  ``nu_ij E_j = nu_ji E_i``
- `rho::float`: density
- `Sit/c::float`: strength in ith direction for tension and compression
- `Sij::float`: strength in ij direction
"""
struct Material{TF}
    E1::TF
    E2::TF
    E3::TF
    G12::TF
    G13::TF
    G23::TF
    nu12::TF
    nu13::TF
    nu23::TF
    rho::TF
    S1t::TF
    S1c::TF
    S2t::TF
    S2c::TF
    S3t::TF
    S3c::TF
    S12::TF
    S13::TF
    S23::TF
end

# strength properties optional
Material(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23, rho) = Material(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23, rho, ones(eltype(E1), 9)...)
Material{TF}(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23, rho) where TF = Material{TF}(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23, rho, ones(TF, 9)...)

Base.eltype(::Material{TF}) where TF = TF
Base.eltype(::Type{Material{TF}}) where TF = TF

Material{TF}(m::Material) where {TF} = Material{TF}(m.E1, m.E2, m.E3, m.G12, m.G13, m.G23, 
m.nu12, m.nu13, m.nu23, m.rho, m.S1t, m.S1c, m.S2t, m.S2c, m.S3t, m.S3c, m.S12, m.S13, m.S23)
Base.convert(::Type{Material{TF}}, m::Material) where {TF} = Material{TF}(m)

"""
    Node(x, y)

A node in the finite element mesh at location x, y.  If assembled in a vector, the vector index corresponds to the node number.

**Arguments**
- `x::float`: x location of node in global coordinate system
- `y::float`: y location of node in global coordinate system
"""
struct Node{TF}
    x::TF
    y::TF
end

Base.eltype(::Node{TF}) where TF = TF
Base.eltype(::Type{Node{TF}}) where TF = TF

Node{TF}(n::Node) where {TF} = Node{TF}(n.x, n.y)
Base.convert(::Type{Node{TF}}, n::Node) where {TF} = Node{TF}(n)

"""
    MeshElement(nodenum, material, theta)

An element in the mesh, consisting of four ordered nodes, a material, and a fiber orientation.

**Arguments**
- `nodenum::Vector{integer}`: a vector of four node numbers corresponding the the four nodes defining this element (vector indices of the nodes). 
    Node order should be counterclockwise starting from the bottom left node using the local coordinate sytem (see figure).
- `material::Material`: material properties of this element
- `theta::float`: fiber orientation
"""
struct MeshElement{VI, TF}
    nodenum::VI
    material::Material{TF}
    theta::TF
end

# for cases wshere strength is not used
MeshElement(nodenum, material, theta) = MeshElement(nodenum, material, theta, Strength(ones(9)...))

Base.eltype(::MeshElement{VI, TF}) where {VI, TF} = TF
Base.eltype(::Type{MeshElement{VI, TF}}) where {VI, TF} = TF

MeshElement{VI,TF}(e::MeshElement) where {VI,TF} = MeshElement{VI,TF}(e.nodenum, e.material, e.theta)
Base.convert(::Type{MeshElement{VI,TF}}, e::MeshElement) where {VI,TF} = MeshElement{VI,TF}(e)




"""
internal cache so allocations happen only once upfront
"""
struct SectionCache{TM, TSM, TSMF, TV, TMF, TAF}  # matrix dual, sparse matrix dual, sparse matrix of floats, vector ints, matrix floats, array floats
    Q::TM
    Ttheta::TSM
    Tbeta::TSM
    Z::TSM
    S::TSMF
    N::TSM
    SZ::TSM
    SN::TSM
    Bksi::TSM
    Beta::TSM
    dNM_dksi::TSM
    dNM_deta::TSM
    BN::TSM
    Ae::TM
    Re::TM
    Ee::TM
    Ce::TM
    Le::TM
    Me::TM
    idx::TV
    A::TM
    R::TM
    E::TSM
    C::TSM
    L::TM
    M::TSM
    X::TM
    Y::TM
    dX::TM
    XY::TM
    X1::TMF
    X2::TMF
    X1dot::TAF
    X2dot::TAF
    B1dot::TAF
    Adot::TAF
end

"""
    initialize_cache(nodes, elements, etype=Float64)

create cache.  set sizes of static matrices, and set sparsity patterns for those that are fixed.

**Arguments**
- `etype::Type`: the element type (typically a float or a dual type)
- `d::Int`: the number of variables you are taking derivatives w.r.t (i.e., number of design variables)
"""
function initialize_cache(nodes, elements, etype=Float64, d=0)

    # create cache
    Q = zeros(etype, 6, 6)
    
    Ttheta = zeros(etype, 6, 6)
    Ttheta[1, 1] = 1.0; Ttheta[1, 4] = 1.0; Ttheta[1, 6] = 1.0
    Ttheta[2, 2] = 1.0
    Ttheta[3, 3] = 1.0; Ttheta[3, 5] = 1.0
    Ttheta[4, 1] = 1.0; Ttheta[4, 4] = 1.0; Ttheta[4, 6] = 1.0
    Ttheta[5, 3] = 1.0; Ttheta[5, 5] = 1.0
    Ttheta[6, 1] = 1.0; Ttheta[6, 4] = 1.0; Ttheta[6, 6] = 1.0
    Ttheta = sparse(Ttheta)
    
    Tbeta = zeros(etype, 6, 6)
    Tbeta[1, 1] = 1.0; Tbeta[1, 2] = 1.0; Tbeta[1, 3] = 1.0
    Tbeta[2, 1] = 1.0; Tbeta[2, 2] = 1.0; Tbeta[2, 3] = 1.0
    Tbeta[3, 1] = 1.0; Tbeta[3, 2] = 1.0; Tbeta[3, 3] = 1.0
    Tbeta[4, 4] = 1.0; Tbeta[4, 5] = 1.0
    Tbeta[5, 4] = 1.0; Tbeta[5, 5] = 1.0
    Tbeta[6, 6] = 1.0
    Tbeta = sparse(Tbeta)
    
    Z = zeros(etype, 3, 6)  # [I zeros(etype, 3, 3)]
    Z[1, 1] = 1.0
    Z[2, 2] = 1.0
    Z[3, 3] = 1.0
    Z[1, 6] = 1.0
    Z[2, 6] = 1.0
    Z[3, 4] = 1.0
    Z[3, 5] = 1.0
    Z = sparse(Z)

    S = [zeros(3, 3); I]
    S = sparse(S)

    N = zeros(etype, 3, 12)
    N[1, 1] = 1.0
    N[2, 2] = 1.0
    N[3, 3] = 1.0
    N[1, 4] = 1.0
    N[2, 5] = 1.0
    N[3, 6] = 1.0
    N[1, 7] = 1.0
    N[2, 8] = 1.0
    N[3, 9] = 1.0
    N[1, 10] = 1.0
    N[2, 11] = 1.0
    N[3, 12] = 1.0
    N = sparse(N)

    SZ = spzeros(etype, 6, 6)
    SN = spzeros(etype, 6, 12)
    
    Bksi = zeros(etype, 6, 3)
    Bksi[1, 1] = 1.0
    Bksi[2, 2] = 1.0
    Bksi[3, 1] = 1.0
    Bksi[3, 2] = 1.0
    Bksi[4, 3] = 1.0
    Bksi[5, 3] = 1.0
    Bksi = sparse(Bksi)
    
    Beta = zeros(etype, 6, 3)
    Beta[1, 1] = 1.0
    Beta[2, 2] = 1.0
    Beta[3, 1] = 1.0
    Beta[3, 2] = 1.0
    Beta[4, 3] = 1.0
    Beta[5, 3] = 1.0
    Beta = sparse(Beta)
    
    dNM_dksi = zeros(etype, 3, 12)
    dNM_dksi[1, 1] = 1.0
    dNM_dksi[2, 2] = 1.0
    dNM_dksi[3, 3] = 1.0
    dNM_dksi[1, 4] = 1.0
    dNM_dksi[2, 5] = 1.0
    dNM_dksi[3, 6] = 1.0
    dNM_dksi[1, 7] = 1.0
    dNM_dksi[2, 8] = 1.0
    dNM_dksi[3, 9] = 1.0
    dNM_dksi[1, 10] = 1.0
    dNM_dksi[2, 11] = 1.0
    dNM_dksi[3, 12] = 1.0
    dNM_dksi = sparse(dNM_dksi)
    
    dNM_deta = zeros(etype, 3, 12)
    dNM_deta[1, 1] = 1.0
    dNM_deta[2, 2] = 1.0
    dNM_deta[3, 3] = 1.0
    dNM_deta[1, 4] = 1.0
    dNM_deta[2, 5] = 1.0
    dNM_deta[3, 6] = 1.0
    dNM_deta[1, 7] = 1.0
    dNM_deta[2, 8] = 1.0
    dNM_deta[3, 9] = 1.0
    dNM_deta[1, 10] = 1.0
    dNM_deta[2, 11] = 1.0
    dNM_deta[3, 12] = 1.0
    dNM_deta = sparse(dNM_deta)
    
    BN = spzeros(etype, 6, 12)

    Ae = zeros(etype, 6, 6)
    Re = zeros(etype, 12, 6)
    Ee = zeros(etype, 12, 12)
    Ce = zeros(etype, 12, 12)
    Le = zeros(etype, 12, 6)
    Me = zeros(etype, 12, 12)

    idx = zeros(Int64, 12)

    # big system matrices
    ne = length(elements) # number of elements
    nn = length(nodes)  # number of nodes
    ndof = 3 * nn  # 3 displacement dof per node

    A = zeros(etype, 6, 6)  # 6 x 6
    R = zeros(etype, ndof, 6)  #  nn*3 x 6
    E = spzeros(etype, ndof, ndof)  # nn*3 x nn*3
    C = spzeros(etype, ndof, ndof)  # nn*3 x nn*3
    L = zeros(etype, ndof, 6)  # nn*3 x 6
    M = spzeros(etype, ndof, ndof)  # nn*3 x nn*3

    # initialize sparsity pattern
    one12 = ones(etype, 12, 12)
    @views for i = 1:ne
        nodenum = elements[i].nodenum
        idx = node2idx(nodenum)

        E[idx, idx] .= one12
        C[idx, idx] .= one12
        M[idx, idx] .= one12
    end
    # reset to zeros
    E .= 0.0
    C .= 0.0
    M .= 0.0

    X = zeros(etype, ndof, 6)
    Y = zeros(etype, 6, 6)
    dX = zeros(etype, ndof, 6)
    XY = zeros(etype, 2*ndof+6, 6)
    
    # ---- used for derivatives ------
    # all of these are always floats
    X1 = zeros(ndof+12, 6)
    X2 = zeros(ndof+12, 6)
    if d != 0
        X1dot = zeros(ndof+12, 6, d)
        X2dot = zeros(ndof+12, 6, d)
        B1dot = zeros(ndof+12, 6, d)
        Adot = zeros(ndof+12, ndof+12, d)
    else
        X1dot = zeros(1, 1, 1)
        X2dot = zeros(1, 1, 1)
        B1dot = zeros(1, 1, 1)
        Adot = zeros(1, 1, 1)
    end

    cache = SectionCache(Q, Ttheta, Tbeta, Z, S, N, SZ, SN, Bksi, Beta, dNM_dksi, dNM_deta, BN, Ae, Re, Ee, Ce, Le, Me, idx, A, R, E, C, L, M, X, Y, dX, XY, X1, X2, X1dot, X2dot, B1dot, Adot)

    return cache
end

"""
Constituitive matrix of this material using the internal ordering.
"""
function stiffness!(material, cache) 
    E1 = material.E1; E2 = material.E2; E3 = material.E3
    nu12 = material.nu12; nu13 = material.nu13; nu23 = material.nu23
    G12 = material.G12; G13 = material.G13; G23 = material.G23

    nu21 = nu12*E2/E1
    nu31 = nu13*E3/E1
    nu32 = nu23*E3/E2
    delta = 1.0 / (1 - nu12*nu21 - nu23*nu32 - nu13*nu31 - 2*nu21*nu32*nu13)

    cache.Q .= 0.0  # reset (needed b/c Q is modifed elsewhere in place so other entires may be nonzero)
    cache.Q[6, 6] = E1*(1 - nu23*nu32)*delta
    cache.Q[1, 1] = E2*(1 - nu13*nu31)*delta
    cache.Q[2, 2] = E3*(1 - nu12*nu21)*delta
    cache.Q[1, 6] = E1*(nu21 + nu31*nu23)*delta
    cache.Q[6, 1] = E1*(nu21 + nu31*nu23)*delta
    cache.Q[2, 6] = E1*(nu31 + nu21*nu32)*delta
    cache.Q[6, 2] = E1*(nu31 + nu21*nu32)*delta
    cache.Q[1, 2] = E2*(nu32 + nu12*nu31)*delta
    cache.Q[2, 1] = E2*(nu32 + nu12*nu31)*delta
    cache.Q[4, 4] = G12
    cache.Q[5, 5] = G13
    cache.Q[3, 3] = G23

    return nothing
end

"""
Rotate constituitive matrix by ply angle
"""
function rotate_ply_to_element!(theta, cache)
    c = cos(theta)
    s = sin(theta)

    cache.Ttheta[1, 1] = c^2
    cache.Ttheta[1, 4] = 2*s*c
    cache.Ttheta[1, 6] = s^2
    cache.Ttheta[2, 2] = 1.0
    cache.Ttheta[3, 3] = c
    cache.Ttheta[3, 5] = s
    cache.Ttheta[4, 1] = -s*c
    cache.Ttheta[4, 4] = c^2 - s^2
    cache.Ttheta[4, 6] = s*c
    cache.Ttheta[5, 3] = -s
    cache.Ttheta[5, 5] = c
    cache.Ttheta[6, 1] = s^2
    cache.Ttheta[6, 4] = -2*s*c
    cache.Ttheta[6, 6] = c^2

    cache.Q .= cache.Ttheta * Symmetric(cache.Q) * cache.Ttheta'
    return nothing
end

"""
Rotate constituitive matrix by element orientation where `c = cos(beta)` and `s = sin(beta)`
"""
function rotate_element_to_beam!(c, s, cache)  # c = cos(beta), s = sin(beta)

    cache.Tbeta[1, 1] = c^2
    cache.Tbeta[1, 2] = s^2
    cache.Tbeta[1, 3] = -2*s*c
    cache.Tbeta[2, 1] = s^2
    cache.Tbeta[2, 2] = c^2
    cache.Tbeta[2, 3] = 2*s*c
    cache.Tbeta[3, 1] = s*c
    cache.Tbeta[3, 2] = -s*c
    cache.Tbeta[3, 3] = c^2-s^2
    cache.Tbeta[4, 4] = c
    cache.Tbeta[4, 5] = -s
    cache.Tbeta[5, 4] = s
    cache.Tbeta[5, 5] = c
    cache.Tbeta[6, 6] = 1.0

    cache.Q .= cache.Tbeta * Symmetric(cache.Q) * cache.Tbeta'
    return nothing
end


"""
Get element constituitive matrix accounting for fiber orientation and element orientation
"""
function elementQ!(material, theta, cbeta, sbeta, cache)
    stiffness!(material, cache)
    rotate_ply_to_element!(theta, cache)
    rotate_element_to_beam!(cbeta, sbeta, cache)

    return nothing
end

"""
Compute the integrand for a single element with a given ksi, eta.
"""
function addelementintegrand!(ksi, eta, element, nodes, cache)

    # shape functions
    N = zeros(4)
    N[1] = 0.25*(1 - ksi)*(1 - eta)
    N[2] = 0.25*(1 + ksi)*(1 - eta)
    N[3] = 0.25*(1 + ksi)*(1 + eta)
    N[4] = 0.25*(1 - ksi)*(1 + eta)
    
    # x, y position
    x = 0.0
    y = 0.0
    for i = 1:4
        x += N[i]*nodes[i].x
        y += N[i]*nodes[i].y
    end

    # orientation (beta)
    xl = 0.5*(nodes[1].x + nodes[4].x)
    yl = 0.5*(nodes[1].y + nodes[4].y)
    xr = 0.5*(nodes[2].x + nodes[3].x)
    yr = 0.5*(nodes[2].y + nodes[3].y)
    dx = xr - xl
    dy = yr - yl
    ds = sqrt(dx^2 + dy^2)
    cbeta = dx/ds
    sbeta = dy/ds
    
    # basic matrices
    # Z = [I [0.0 0 -y; 0 0 x; y -x 0]]  # translation and cross product
    cache.Z[1, 6] = -y
    cache.Z[2, 6] = x
    cache.Z[3, 4] = y
    cache.Z[3, 5] = -x
    # S = [zeros(3, 3); I]
    elementQ!(element.material, element.theta, cbeta, sbeta, cache)
    cache.N[1, 1] = N[1]
    cache.N[2, 2] = N[1]
    cache.N[3, 3] = N[1]
    cache.N[1, 4] = N[2]
    cache.N[2, 5] = N[2]
    cache.N[3, 6] = N[2]
    cache.N[1, 7] = N[3]
    cache.N[2, 8] = N[3]
    cache.N[3, 9] = N[3]
    cache.N[1, 10] = N[4]
    cache.N[2, 11] = N[4]
    cache.N[3, 12] = N[4]
    cache.SZ .= cache.S * cache.Z
    cache.SN .= cache.S * cache.N

    # derivatives of shape functions
    dN_dksi = zeros(4)
    dN_dksi[1] = -0.25*(1 - eta)
    dN_dksi[2] = 0.25*(1 - eta)
    dN_dksi[3] = 0.25*(1 + eta)
    dN_dksi[4] = -0.25*(1 + eta)

    dN_deta = zeros(4)
    dN_deta[1] = -0.25*(1 - ksi)
    dN_deta[2] = -0.25*(1 + ksi)
    dN_deta[3] = 0.25*(1 + ksi)
    dN_deta[4] = 0.25*(1 - ksi)

    # Jacobian
    dx_dksi = 0.0
    dx_deta = 0.0
    dy_dksi = 0.0
    dy_deta = 0.0
    for i = 1:4
        dx_dksi += dN_dksi[i]*nodes[i].x
        dx_deta += dN_deta[i]*nodes[i].x
        dy_dksi += dN_dksi[i]*nodes[i].y
        dy_deta += dN_deta[i]*nodes[i].y
    end

    J = [dx_dksi dy_dksi;
         dx_deta dy_deta]
    detJ = det(J)
    Jinv = [dy_deta -dy_dksi;
           -dx_deta dx_dksi] / detJ

    # BN matrix
    cache.Bksi[1, 1] = Jinv[1, 1]
    cache.Bksi[2, 2] = Jinv[2, 1]
    cache.Bksi[3, 1] = Jinv[2, 1]
    cache.Bksi[3, 2] = Jinv[1, 1]
    cache.Bksi[4, 3] = Jinv[1, 1]
    cache.Bksi[5, 3] = Jinv[2, 1]

    cache.Beta[1, 1] = Jinv[1, 2]
    cache.Beta[2, 2] = Jinv[2, 2]
    cache.Beta[3, 1] = Jinv[2, 2]
    cache.Beta[3, 2] = Jinv[1, 2]
    cache.Beta[4, 3] = Jinv[1, 2]
    cache.Beta[5, 3] = Jinv[2, 2]

    # dNM_dksi = [dN_dksi[1]*Matrix(1.0I, 3, 3) dN_dksi[2]*I dN_dksi[3]*I dN_dksi[4]*I]
    cache.dNM_dksi[1, 1] = dN_dksi[1]
    cache.dNM_dksi[2, 2] = dN_dksi[1]
    cache.dNM_dksi[3, 3] = dN_dksi[1]
    cache.dNM_dksi[1, 4] = dN_dksi[2]
    cache.dNM_dksi[2, 5] = dN_dksi[2]
    cache.dNM_dksi[3, 6] = dN_dksi[2]
    cache.dNM_dksi[1, 7] = dN_dksi[3]
    cache.dNM_dksi[2, 8] = dN_dksi[3]
    cache.dNM_dksi[3, 9] = dN_dksi[3]
    cache.dNM_dksi[1, 10] = dN_dksi[4]
    cache.dNM_dksi[2, 11] = dN_dksi[4]
    cache.dNM_dksi[3, 12] = dN_dksi[4]
    # dNM_deta = [dN_deta[1]*Matrix(1.0I, 3, 3) dN_deta[2]*I dN_deta[3]*I dN_deta[4]*I]
    cache.dNM_deta[1, 1] = dN_deta[1]
    cache.dNM_deta[2, 2] = dN_deta[1]
    cache.dNM_deta[3, 3] = dN_deta[1]
    cache.dNM_deta[1, 4] = dN_deta[2]
    cache.dNM_deta[2, 5] = dN_deta[2]
    cache.dNM_deta[3, 6] = dN_deta[2]
    cache.dNM_deta[1, 7] = dN_deta[3]
    cache.dNM_deta[2, 8] = dN_deta[3]
    cache.dNM_deta[3, 9] = dN_deta[3]
    cache.dNM_deta[1, 10] = dN_deta[4]
    cache.dNM_deta[2, 11] = dN_deta[4]
    cache.dNM_deta[3, 12] = dN_deta[4]

    cache.BN .= sparse(cache.Bksi) * sparse(cache.dNM_dksi) + sparse(cache.Beta) * sparse(cache.dNM_deta)
    
    # integrands
    cache.Ae .+= cache.SZ' * cache.Q * cache.SZ * detJ
    cache.Re .+= cache.BN' * cache.Q * cache.SZ * detJ
    cache.Ee .+= cache.BN' * cache.Q * cache.BN * detJ
    cache.Ce .+= cache.BN' * cache.Q * cache.SN * detJ  # use Giavottoa definition
    cache.Le .+= cache.SN' * cache.Q * cache.SZ * detJ
    cache.Me .+= cache.SN' * cache.Q * cache.SN * detJ

    return cbeta, sbeta
end

"""
2-point Gauss quadrature of one element
"""
function submatrix!(element, nodes, cache)

    # initialize
    cache.Ae .= 0.0
    cache.Re .= 0.0
    cache.Ee .= 0.0
    cache.Ce .= 0.0
    cache.Le .= 0.0
    cache.Me .= 0.0

    # 2-point Gauss quadrature in both dimensions
    ksi = [1.0/sqrt(3), 1.0/sqrt(3), -1.0/sqrt(3), -1.0/sqrt(3)]
    eta = [1.0/sqrt(3), -1.0/sqrt(3), 1.0/sqrt(3), -1.0/sqrt(3)]
    
    for i = 1:4
        addelementintegrand!(ksi[i], eta[i], element, nodes, cache)
    end
    
    return nothing
end

"""
Convenience function to map node numbers to locations in global matrix.
"""
function node2idx!(nodenums, cache)
    nn = 4
    # idx = Vector{Int64}(undef, nn*3)
    for i = 1:nn
        cache.idx[((i-1)*3+1):i*3] = ((nodenums[i]-1)*3+1):nodenums[i]*3
    end
    return nothing
end

function node2idx(nodenums)
    nn = 4
    idx = Vector{Int64}(undef, nn*3)
    for i = 1:nn
        idx[((i-1)*3+1):i*3] = ((nodenums[i]-1)*3+1):nodenums[i]*3
    end
    return idx
end


"""
Reorder stiffness or compliance matrix from internal order to GXBeam order
"""
function reorder(K)  # reorder to GXBeam format
    idx = [3, 1, 2, 6, 4, 5]
    return K[idx, idx]
end

"""
pull out linear solve so can overload it with analytic derivatives
solves Ax = b but takes in factorization of A (AF).  Only needs A for the 
overloaded portion but must keep it here to maintain function signature. 
uses cache to avoid allocations
"""
function linearsolve1(A, B1, AF, cache)

    _, n = size(B1)
    for j = 1:n
        cache.X1[:, j] = AF \ B1[:, j]
    end

    return cache.X1
end

"""
pull out linear solve so can overload it with analytic derivatives
solves AM x = B (after factorizing it).  uses cache to avoid allocations
Return matrix factorization (for later reuse) in addition to solution
"""
function linearsolve2(AM, B2, cache)
    AM = factorize(Symmetric(sparse(AM)))

    _, n = size(B2)
    for j = 1:n
        cache.X2[:, j] = AM \ B2[:, j]
    end

    return AM, cache.X2
end

"""
Overloaded version to propagate analytic derivatives with forwarddiff
"""
function linearsolve1(A::SparseMatrixCSC{<:ForwardDiff.Dual{T}}, B1, AF, cache) where {T}

     # extract primal values
    #  Av = ForwardDiff.value.(A)
    B1v = ForwardDiff.value.(B1)
 
     # linear solve
    #  Av = factorize(Av)
    #  x1v = zeros(size(B1v))
    _, n = size(B1v)
    for j = 1:n
        cache.X1[:, j] = AF \ B1v[:, j]
    end
 
     # extract dual values
    ap = ForwardDiff.partials.(A)
    m, n = size(A)
    d = length(ap[1, 1])
    @views for i = 1:m
        for j = 1:n
            cache.Adot[i, j, :] .= ap[i, j].values
        end
    end    
     
    bp = ForwardDiff.partials.(B1)
    m, n = size(B1v)
    @views for i = 1:m
        for j = 1:n
            cache.B1dot[i, j, :] .= bp[i, j].values
        end
    end
 
     # analytic derivative of linear solve
    for i = 1:d
        for j = 1:n
            cache.X1dot[:, j, i] = AF \ (view(cache.B1dot, :, j, i) - view(cache.Adot, :, :, i) * view(cache.X1, :, j))
        end
    end
 
     # repack in dual
    X1D = ForwardDiff.Dual{T}.(cache.X1, ForwardDiff.Partials.(Tuple.(view(cache.X1dot, i, j, :) for i = 1:m, j = 1:n)))
    
    return X1D

end

"""
Overloaded version to propagate analytic derivatives with forwarddiff
"""
function linearsolve2(A::SparseMatrixCSC{<:ForwardDiff.Dual{T}}, B2, cache) where {T}

    # extract primal values
    Av = ForwardDiff.value.(A)
    B2v = B2  # no partials

    # linear solve
    AF = factorize(Symmetric(sparse(Av)))
    _, n = size(B2v)
    for j = 1:n
        cache.X2[:, j] = AF \ B2v[:, j]
    end

    # extract dual values
    ap = ForwardDiff.partials.(A)
    m, n = size(Av)
    d = length(ap[1, 1])
    @views for i = 1:m
        for j = 1:n
           cache.Adot[i, j, :] .= ap[i, j].values
        end
    end    

    m, n = size(B2v)
    
    # analytic derivative of linear solve
    for i = 1:d
        for j = 1:n
            cache.X2dot[:, j, i] = AF \ (- view(cache.Adot, :, :, i) * view(cache.X2, :, j))
        end
    end

    # repack in dual
   X2D = ForwardDiff.Dual{T}.(cache.X2, ForwardDiff.Partials.(Tuple.(view(cache.X2dot, i, j, :) for i = 1:m, j = 1:n)))

   return AF, X2D

end


"""
    compliance_matrix(nodes, elements; cache=initialize_cache(nodes, elements), gxbeam_order=true)

Compute compliance matrix given the finite element mesh described by nodes and elements.

**Arguments**
- `nodes::Vector{Node{TF}}`: all the nodes in the mesh
- `elements::Vector{MeshElement{VI, TF}}`: all the elements in the mesh
- `cache::SectionCache`: if number of nodes, number of elements, and connectivity of mesh stays the same (and you will be repeating calls)
    then you can should initialize cache yourself and pass in so you don't have to keep reconstructing it.
- `gxbeam_order::Bool`: true if output compliance matrix should be in GXBeam order or internal ordering

**Returns**
- `S::Matrix`: compliance matrix (about the shear center as long as gxbeam_order = true)
- `sc::Vector{float}`: x, y location of shear center
    (location where a transverse/shear force will not produce any torsion, i.e., beam will not twist)
- `tc::Vector{float}`: x, y location of tension center, aka elastic center, aka centroid 
    (location where an axial force will not produce any bending, i.e., beam will remain straight)
"""
function compliance_matrix(nodes, elements; cache=initialize_cache(nodes, elements), gxbeam_order=true)

    TF = promote_type(eltype(eltype(nodes)), eltype(eltype(elements)))

    # initialize
    ne = length(elements) # number of elements
    nn = length(nodes)  # number of nodes
    ndof = 3 * nn  # 3 displacement dof per node

    # place element matrices in global matrices (scatter)
    cache.A .= 0.0
    cache.R .= 0.0
    cache.E .= 0.0
    cache.C .= 0.0
    cache.L .= 0.0
    cache.M .= 0.0
    @views for i = 1:ne
        nodenum = elements[i].nodenum
        submatrix!(elements[i], nodes[nodenum], cache)
        node2idx!(nodenum, cache)

        cache.A .+= cache.Ae
        cache.R[cache.idx, :] .+= cache.Re
        cache.E[cache.idx, cache.idx] .+= cache.Ee
        cache.C[cache.idx, cache.idx] .+= cache.Ce
        cache.L[cache.idx, :] .+= cache.Le
        cache.M[cache.idx, cache.idx] .+= cache.Me
    end
    A = cache.A
    R = cache.R
    E = cache.E
    C = cache.C
    L = cache.L
    M = cache.M

    # assemble displacement constraint matrix
    DT = spzeros(TF, 6, ndof)
    for i = 1:nn
        s = 3*(i-1)
        DT[1, s+1] = 1.0
        DT[2, s+2] = 1.0
        DT[3, s+3] = 1.0
        DT[4, s+3] = nodes[i].y
        DT[5, s+3] = -nodes[i].x
        DT[6, s+1] = -nodes[i].y
        DT[6, s+2] = nodes[i].x
    end
    D = sparse(DT')

    # Tr matrix
    Tr = spzeros(6, 6)
    Tr[1, 5] = -1.0
    Tr[2, 4] = 1.0

    # solve first linear system
    AM = [E R D;
        R' A zeros(6, 6);
        DT zeros(6, 12)]
    # AM = factorize(Symmetric(sparse(AM)))
    
    B2 = sparse([zeros(ndof, 6); Tr'; zeros(6, 6)])
    AF, X2 = linearsolve2(AM, B2, cache)
    cache.dX .= view(X2, 1:ndof, :)
    # dY = X2[ndof+1:ndof+6, :]

    # solve second linear system
    Bsub1 = [C'-C  L;  # NOTE: error in manual should be C' - C
            -L' zeros(6, 6);  # NOTE: sign fix, error in BECAS documentation
            zeros(6, ndof+6)]
    Bsub2 = [zeros(ndof, 6); I; zeros(6, 6)]
    B1 = sparse(Bsub1*X2[1:end-6, :] + Bsub2)
    X1 = linearsolve1(AM, B1, AF, cache)
    cache.X .= view(X1, 1:ndof, :)
    cache.Y .= view(X1, ndof+1:ndof+6, :)

    # compliance matrix
    cache.XY[1:ndof, :] .= cache.X
    cache.XY[ndof+1:2*ndof, :] .= cache.dX
    cache.XY[2*ndof+1:end, :] .= cache.Y
    S = cache.XY'*[E C R; C' M L; R' L' A]*cache.XY

    xs = -S[6, 2]/S[6, 6]
    ys = S[6, 1]/S[6, 6]
    xt = (S[4, 4]*S[5, 3] - S[4, 5]*S[4, 3])/(S[4, 4]*S[5, 5] - S[4, 5]^2)
    yt = (-S[4, 3]*S[5, 5] + S[4, 5]*S[5, 3])/(S[4, 4]*S[5, 5] - S[4, 5]^2) 
    sc = [xs, ys]
    tc = [xt, yt]


    if gxbeam_order
        # change ordering to match gxbeam
        S = reorder(S)

        # move properties to be about the shear center (note x->y, y->z b.c. x is axial in derivation)
        P = [0.0 -ys xs
             ys   0   0
            -xs   0   0]
        Hinv = [I transpose(P); zeros(3, 3) I]
        HinvT = [I zeros(3, 3); P I]

        S = Hinv * S * HinvT
    end


    return S, sc, tc
end

"""
compute area and centroid of an element specified by its four nodes
"""
function area_and_centroid_of_element(node)

    # shoelace formula for area
    A = 0.5 * (
        node[1].x * node[2].y - node[2].x * node[1].y + 
        node[2].x * node[3].y - node[3].x * node[2].y + 
        node[3].x * node[4].y - node[4].x * node[3].y + 
        node[4].x * node[1].y - node[1].x * node[4].y)

    # centroid of element
    xc = (node[1].x + node[2].x + node[3].x + node[4].x)/4
    yc = (node[1].y + node[2].y + node[3].y + node[4].y)/4

    return A, xc, yc
end


"""
    mass_matrix(nodes, elements)

Compute mass matrix for the structure using GXBeam ordering.
    
**Returns**
- `M::Matrix`: mass matrix
- `mc::Vector{float}`: x, y location of mass center
"""
function mass_matrix(nodes, elements)

    # --- find total mass and center of mass -----
    m = 0.0
    xm = 0.0
    ym = 0.0

    for elem in elements
        # extract nodes and density, compute area and centroid
        node = nodes[elem.nodenum]
        rho = elem.material.rho
        A, xc, yc = area_and_centroid_of_element(node)

        # mass and (numerator of) center of mass
        dm = rho * A
        m += dm
        xm += xc * dm
        ym += yc * dm
    end

    # center of mass
    xm /= m
    ym /= m

    # ----------- compute moments of inertia ---------
    Ixx = 0.0
    Iyy = 0.0
    Ixy = 0.0

    for elem in elements
        # extract nodes and density, compute area and centroid
        node = nodes[elem.nodenum]
        rho = elem.material.rho
        A, xc, yc = area_and_centroid_of_element(node)

        Ixx += (yc - ym)^2 * rho * A
        Iyy += (xc - xm)^2 * rho * A
        Ixy += (xc - xm) * (yc - ym) * rho * A
    end

    M = Symmetric([
        m 0.0 0 0 m*ym -m*xm
        0 m 0 -m*ym 0 0
        0 0 m m*xm 0 0
        0 -m*ym m*xm Ixx+Iyy 0 0
        m*ym 0 0 0 Ixx -Ixy
        -m*xm 0 0 0 -Ixy Iyy
    ])

    return M, [xm, ym]
end

"""
    plotmesh(nodes, elements, pyplot; plotnumbers=false)

plot nodes and elements for a quick visualization.
Need to pass in a PyPlot object as PyPlot is not loaded by this package.
"""
function plotmesh(nodes, elements, pyplot; plotnumbers=false)
    ne = length(elements)
    
    for i = 1:ne
        node = nodes[elements[i].nodenum]
        for i = 1:4
            iplus = i+1
            if iplus == 5
                iplus = 1 
            end
            pyplot.plot([node[i].x, node[iplus].x], [node[i].y, node[iplus].y], "k")
        end
        if plotnumbers
            barx = sum([n.x/4 for n in node])
            bary = sum([n.y/4 for n in node])
            pyplot.text(barx, bary, string(i), color="r")
        end
    end
    if plotnumbers
        nn = length(nodes)
        for i = 1:nn
            pyplot.text(nodes[i].x, nodes[i].y, string(i))
        end
    end
end


"""
    strain_recovery(F, M, nodes, elements, cache)

Compute stresses and strains at each element in cross section.

# Arguments
- `F::Vector(3)`: force at this cross section in x, y, z directions
- `M::Vector(3)`: moment at this cross section in x, y, z directions
- `nodes::Vector{Node{TF}}`: all the nodes in the mesh
- `elements::Vector{MeshElement{VI, TF}}`: all the elements in the mesh
- `cache::SectionCache`: needs to reuse data from the compliance solve 
    (thus must initialize cache and pass it to both compliance and this function)

# Returns
- `strain_b::Vector(6, ne)`: strains in beam coordinate system for each element. order: xx, yy, zz, xy, xz, yz
- `stress_b::Vector(6, ne)`: stresses in beam coordinate system for each element. order: xx, yy, zz, xy, xz, yz
- `strain_p::Vector(6, ne)`: strains in ply coordinate system for each element. order: 11, 22, 33, 12, 13, 23
- `stress_p::Vector(6, ne)`: stresses in ply coordinate system for each element. order: 11, 22, 33, 12, 13, 23
"""
function strain_recovery(F, M, nodes, elements, cache)
    
    # initialize
    T = promote_type(eltype(F), eltype(M), eltype(cache.X))
    ne = length(elements)
    Xe = Matrix{T}(undef, 12, 6)
    dXe = Matrix{T}(undef, 12, 6)

    # beam and ply c.s.
    epsilon_b = Matrix{T}(undef, 6, ne)
    sigma_b = Matrix{T}(undef, 6, ne)
    epsilon_p = Matrix{T}(undef, 6, ne)
    sigma_p = Matrix{T}(undef, 6, ne)

    # concatenate forces/moments
    theta = [F; M]

    # save reordering index
    idx_b = [1, 2, 6, 3, 4, 5]   # xx, yy, zz, xy, xz, yz
    idx_p = [6, 1, 2, 4, 5, 3]   # 11, 22, 33, 12, 13, 23

    # iterate over elements
    @views for i = 1:ne

        # analyze this element
        elem = elements[i]

        # compute submatrices SZ, BN, SN (evaluated at center of element)
        nodenum = elem.nodenum
        node = nodes[nodenum]
        cbeta, sbeta = addelementintegrand!(0.0, 0.0, elem, node, cache)  # this computes extra stuff we don't need so could be factored out if desired

        # extract part of solution corresponding to this element
        node2idx!(nodenum, cache)
        Xe .= cache.X[cache.idx, :]
        dXe .= cache.dX[cache.idx, :]

        # solve for strains in beam c.s.
        epsilon_b[:, i] .= cache.SZ*cache.Y*theta + cache.BN*Xe*theta + cache.SN*dXe*theta

        # compute corresponding strains
        elementQ!(elem.material, elem.theta, cbeta, sbeta, cache)
        sigma_b[:, i] .= cache.Q * epsilon_b[:, i]

        # compute transformation matrices and rotate strains to ply c.s.
        rotate_ply_to_element!(elem.theta, cache)
        rotate_element_to_beam!(cbeta, sbeta, cache)
        epsilon_p[:, i] .= cache.Ttheta' * cache.Tbeta' * epsilon_b[:, i]

        # compute Q matrix for element and corresponding stress
        stiffness!(elem.material, cache)
        sigma_p[:, i] .= cache.Q * epsilon_p[:, i]

        # reorder to a more conventional order
        epsilon_b[:, i] .= epsilon_b[idx_b, i]
        sigma_b[:, i] .= sigma_b[idx_b, i]
        epsilon_p[:, i] .= epsilon_p[idx_p, i]
        sigma_p[:, i] .= sigma_p[idx_p, i]
    end

    return epsilon_b, sigma_b, epsilon_p, sigma_p
end


"""
    plotsoln(nodes, elements, soln, pyplot)

plot stress/strain on mesh
soln could be any vector that is of length # of elements, e.g., sigma_b[3, :]
Need to pass in a PyPlot object as PyPlot is not loaded by this package.
"""
function plotsoln(nodes, elements, soln, pyplot)
    ne = length(elements)
    nn = length(nodes)
    
    # extract node points
    xpts = zeros(nn)
    ypts = zeros(nn)
    for i = 1:nn
        xpts[i] = nodes[i].x
        ypts[i] = nodes[i].y
    end
    
    # split quads into trianagles
    triangles = zeros(Int64, ne*2, 3)
    trisol = zeros(ne*2)  

    for i = 1:ne
        nnum = elements[i].nodenum

        triangles[i*2-1, :] = nnum[1:3] .- 1
        triangles[i*2, :] = [nnum[1], nnum[3], nnum[4]] .- 1

        # same solution on both triangles (same quad element)
        trisol[2*i-1] = soln[i]
        trisol[2*i] = soln[i]
    end

    pyplot.tripcolor(xpts, ypts, trisol, triangles=triangles)
end


# function tsai_hill(sigma, strength)
    
#     (; S1t, S1c, S2t, S2c, S3t, S3c, S12, S13, S23) = strength
    
#     _, ne = size(sigma)
#     T = eltype(sigma)
#     failure = Vector{T}(undef, n)  # fails if > 1
#     s = Vector{T}(undef, 6)

#     for i = 1:ne
#         s .= sigma[:, i]

#         if s[1] >= 0.0
#             S1 = S1t
#         else
#             S1 = S1c
#         end
#         if s[2] >= 0.0
#             S2 = S2t
#         else
#             S2 = S2c
#         end
#         failure[i] = s[1]^2/S1^2 + s[2]^2/S2^2 + s[4]^2/S12^2 - s[1]*s[2]/S1^2
#     end

#     return failure
# end

"""
    tsai_wu(stress_p, elements)

Tsai Wu failure criteria

# Arguments
- `stress_p::vector(6, ne)`: stresses in ply coordinate system
- `elements::Vector{MeshElement{VI, TF}}`: all the elements in the mesh

# Returns
- `failure::vector(ne)`: tsai-wu failure criteria for each element.  fails if >= 1
"""
function tsai_wu(stress_p, elements)

    ne = length(elements)
    T = eltype(stress_p)
    failure = Vector{T}(undef, ne)  # fails if > 1
    s = Vector{T}(undef, 6)

    @views for i = 1:ne
        m = elements[i].material
        s .= stress_p[:, i]
        failure[i] = s[1]^2/(m.S1t*m.S1c) + 
                     s[2]^2/(m.S2t*m.S2c) +
                     s[3]^2/(m.S3t*m.S3c) +
                     s[4]^2/m.S12^2 + 
                     s[5]^2/m.S13^2 + 
                     s[6]^2/m.S23^2 + 
                     s[1]*(1/m.S1t - 1/m.S1c) + 
                     s[2]*(1/m.S2t - 1/m.S2c) + 
                     s[3]*(1/m.S3t - 1/m.S3c) - 
                     s[1]*s[2]/sqrt(m.S1t*m.S1c*m.S2t*m.S2c) -
                     s[1]*s[3]/sqrt(m.S1t*m.S1c*m.S3t*m.S3c) -
                     s[2]*s[3]/sqrt(m.S2t*m.S2c*m.S3t*m.S3c)
    end
        
    return failure
end