"""Techniques for subsampling a mesh."""
import numpy as np
from scipy.linalg import qr, svd, lu, det, cholesky, lstsq
from copy import deepcopy
def get_qr_column_pivoting(Ao, number_of_subsamples):
    """
    Pivoted QR factorization, where the pivots are used as a heuristic for subsampling.
    """
    A = deepcopy(Ao)
    __, __, pvec = qr(A.T, pivoting=True)
    z = pvec[0:number_of_subsamples]
    return z
def get_svd_subset_selection(Ao, number_of_subsamples):
    """
    Singular value decomposition and pivoted QR factorization, where the pivots
    are used as a heuristic for subsampling.
    """
    A = deepcopy(Ao)
    __, __, V = svd(A.T)
    __, __, pvec = qr(V[:, 0:number_of_subsamples].T , pivoting=True )
    z = pvec[0:number_of_subsamples]
    return z
def get_newton_determinant_maximization(Ao, number_of_subsamples):
    """
    A convex relaxation technique for determinant maximization---akin to optimal experiment of
    design (D). Based on the work of Joshi and Boyd [1].

    **References**
        1. Joshi, S., Boyd, S., (2009) Sensor Selection via Convex Optimization. IEEE Transactions on Signal Processing, 57(2). `Paper <https://ieeexplore.ieee.org/document/4663892>`__

    """
    A = deepcopy(Ao)
    maxiter = 50
    n_tol = 1e-12
    gap = 1.005s

    # For backtracking line search parameters
    alpha = 0.01
    beta = 0.5

    # Assuming the input matrix is an np.matrix()
    m, n = A.shape
    if m < n:
        raise(ValueError, 'maxdet(): requires the number of columns to be greater than the number of rows!')
    z = np.ones((m, 1)) * float(number_of_subsamples)/float(m)
    g = np.zeros((m, 1))
    ones_m = np.ones((m, 1))
    ones_m_transpose = np.ones((1, m))
    kappa = np.log(gap) * n/m

    # Objective function
    Z = diag(z)
    fz = -np.log(np.linalg.det(A.T * Z * A)) - kappa * np.sum(np.log(z) + np.log(1.0 - z))

    # Optimization loop!
    for i in range(0, maxiter) :
        Z = diag(z)
        W = np.linalg.inv(A.T * Z * A)
        V = A * W * A.T
        vo = np.matrix(np.diag(V))
        vo = vo.T

        # define some z operations
        one_by_z = ones_m / z
        one_by_one_minus_z = ones_m / (ones_m - z)
        one_by_z2 = ones_m / z**2
        one_by_one_minus_z2 = ones_m / (ones_m - z)**2
        g = -vo- kappa * (one_by_z - one_by_one_minus_z)
        H = np.multiply(V, V) + kappa * diag( one_by_z2 + one_by_one_minus_z2)

        # Textbook Newton's method -- compute inverse of Hessian
        R = np.matrix(cholesky(H) )
        u = lstsq(R.T, g)
        Hinvg = lstsq(R, u[0])
        Hinvg = Hinvg[0]
        v = lstsq(R.T, ones_m)
        Hinv1 = lstsq(R, v[0])
        Hinv1 = Hinv1[0]
        dz = -Hinvg + (np.dot( ones_m_transpose , Hinvg ) / np.dot(ones_m_transpose , Hinv1)) * Hinv1


        deczi = indices(dz, lambda x: x < 0)
        inczi = indices(dz, lambda x: x > 0)
        a1 = 0.99* -z[deczi, 0] / dz[deczi, 0]
        a2 = (1 - z[inczi, 0] )/dz[inczi, 0]
        s = np.min(np.vstack([1.0, np.vstack(a1), np.vstack(a2) ] )  )
        flag = 1

        while flag == 1:
            zp = z + s*dz
            Zp = diag(zp)
            fzp = -np.log(np.linalg.det(A.T * Zp * A) ) - kappa * np.sum(np.log(zp) + np.log(1 - zp)  )
            const = fz + alpha * s * g.T * dz
            if fzp <= const[0,0]:
                flag = 2
            if flag != 2:
                s = beta * s
        z = zp
        fz = fzp
        sig = -g.T * dz * 0.5
        if( sig[0,0] <= n_tol):
            break
        zsort = np.sort(z, axis=0)
        thres = zsort[m - number_of_subsamples - 1]
        zhat, not_used = find(z, thres)

    zsort = np.sort(z, axis=0)
    thres = zsort[m - number_of_subsamples - 1]
    zhat, not_used = find(z, thres)
    p, q = zhat.shape
    Zhat = diag(zhat)
    L = np.log(np.linalg.det(A.T * Zhat  * A))
    ztilde  = z
    Utilde = np.log(np.linalg.det(A.T * diag(z) * A))  + 2 * m * kappa
    z = __binary2indices(zhat)
def __binary2indices(zhat):
    """
    Simple utility that converts a binary array into one with indices!
    """
    pvec = []
    m, n = zhat.shape
    for i in range(0, m):
        if(zhat[i,0] == 1):
            pvec.append(i)
    return pvec
def __CG_solve(A, b, max_iters, tol):
    """
    Solves Ax = b iteratively using conjugate gradient, assuming A is a symmetric positive definite matrix.

    :param numpy-matrix A:
        The matrix.
    :param numpy-array b:
        The right hand side column vector.
    :param int max_iters:
        Maximum number of iterations for the conjugate gradient algorithm.
    :param double tol:
        Tolerance for cut-off.

    """
    if not(np.all(np.linalg.eigvals(A) > 0)):
        raise ValueError('A is not symmetric positive definite.')
    n = A.shape[0]
    b = b.astype(np.float64)
    b = b.reshape((len(b),1))


    #Initialization
    x = np.zeros((n,1))
    r = b.copy()

    d = r.copy()
    iterations = 0
    delta = sum(r**2)
    delta_0 = sum(b**2)
    bestx = x.copy()
    bestres = np.sqrt(delta/delta_0)
    residual = np.sqrt(delta / delta_0)

    while (iterations < max_iters) and (delta > (tol**2) * delta_0):

        q = np.dot(A,d)
        alpha = delta / sum(d * q)

        x += alpha * d
        r -= alpha * q
        new_delta = sum(r**2)
        beta = new_delta / delta
        d = r + beta * d

        if np.sqrt(delta/delta_0) < bestres:
            bestx = x.copy()
            bestres = np.sqrt(delta/delta_0)

        delta = new_delta
        residual = np.sqrt(delta / delta_0)
        iterations += 1

    return x.flatten(), residual, iterations
def __bp(A, b, x0 = None, cgtol = None, cgmaxiter = None, pdtol = None, pdmaxiter = None, verbose = False, use_CG = False):
    """
    Solving the noiseless basis pursuit problem.

    :param numpy-matrix A:
        The matrix.
    :param numpy-array b:
        The right hand side column vector.
    :param numpy-array x0:
        Initial solution  if not provided the least norm solution is used.
    :param double cgtol:
        Tolerance for conjugate gradients.
    :param int cgmaxiter:
        Maximum number of iterations for the conjugate gradient method.
    """

    # Free parameters
    alpha = .01
    beta = .5
    mu = 10

    if cgtol is None:
        cgtol = 1e-5
    if cgmaxiter is None:
        cgmaxiter = 200
    if pdtol is None:
        pdtol = 1e-3
    if pdmaxiter is None:
        pdmaxiter = 50

    b = b.flatten()

    # Find initial solution, if none provided or provided but infeasible
    if not(x0 is None):
        x0 = x0.flatten()
        if np.linalg.norm(np.dot(A,x0).flatten() - b)/np.linalg.norm(b) > cgtol:
            if use_CG:
                q, cgres, cgiter =  __CG_solve(np.dot(A,A.T), b, cgmaxiter, cgtol)
            else:
                q = np.linalg.solve(np.dot(A,A.T), b)
                cgres = np.linalg.norm(np.dot(np.dot(A,A.T), q).flatten() - b.flatten()) / np.linalg.norm(b)
                cgiter = -1
            if cgres > 0.5:

                raise ValueError('AA^T is too ill conditioned. Cannot find starting point.')
                xp = x0.copy()
                return xp
            x0 = np.dot(A.T, q)
    else:

        if use_CG:
            q, cgres, cgiter =  __CG_solve(np.dot(A,A.T), b, cgmaxiter, cgtol)
        else:
            q = np.linalg.solve(np.dot(A,A.T), b)
            cgres = np.linalg.norm(np.dot(np.dot(A,A.T), q).flatten() - b.flatten()) / np.linalg.norm(b)
            cgiter = -1
        if cgres > 0.5:
            raise ValueError('AA^T is too ill conditioned. Cannot find starting point.')
            return
        x0 = np.dot(A.T, q)

    N = len(x0)
    gradf0 = np.hstack([np.zeros((N,),dtype = np.float64), np.ones((N,),dtype = np.float64)])

    x = x0.flatten()
    u = 0.95*np.abs(x0) + 0.10*np.max(np.abs(x0))

    # First iteration
    fu1 = x - u
    fu2 = -x - u
    lam_u1 = -1.0 / fu1
    lam_u2 = -1.0 / fu2
    #initialize nu (v) to be -A*(lam_u1 - lam_u2)
    v = np.dot(-A, (lam_u1-lam_u2)).flatten()

    sdg = -(sum(fu1*lam_u1) + sum(fu2*lam_u2))
    tau = mu * 2.0 * N / sdg

    #Calculate the residuals
    Atv = np.dot(A.T, v).flatten()
    r_pri = np.dot(A, x).flatten() - b
    r_cent = np.hstack([-lam_u1 * fu1, -lam_u2 * fu2]) - 1.0/tau
    r_dual = gradf0 + np.hstack([lam_u1 - lam_u2, -lam_u1 - lam_u2]) + np.hstack([Atv, np.zeros(N)])
    resnorm = np.linalg.norm(np.hstack([r_pri, r_cent, r_dual]))

    pditer = 0
    done = (sdg < pdtol) or (pditer >= pdmaxiter)

    while not(done):
        pditer += 1

        w1 = -1.0/tau*(-1.0/fu1 + 1.0/fu2) - Atv
        w2 = -1.0 - 1.0/tau*(1.0/fu1 + 1.0/fu2)
        w3 = -r_pri
        sig1 = -lam_u1/fu1 - lam_u2/fu2
        sig2 = lam_u1/fu1 - lam_u2/fu2
        sigx = sig1 - sig2**2.0/sig1

        w1p = -w3 + np.dot(A,(w1/sigx - w2*sig2/(sigx*sig1))).flatten()
        inv_sigx = 1.0/sigx
        H11p = np.dot(np.dot(A,(np.diag(inv_sigx.reshape(len(inv_sigx))))),A.T)
        if use_CG:
            dv, cgres, cgiter =  __CG_solve(H11p, w1p, cgmaxiter, cgtol)
        else:
            dv = np.linalg.solve(H11p, w1p)
            cgres = np.linalg.norm(np.dot(H11p, dv).flatten() - w1p.flatten()) / np.linalg.norm(w1p)
            cgiter = -1
        if cgres > .5:
            print(cgres)
            raise ValueError('Matrix ill-conditioned.  Returning previous iterate.')
            xp = x.copy()
            return xp
        dx = (w1 - w2*sig2/sig1 - np.dot(A.T,dv))/sigx
        Adx = np.dot(A,dx)
        Atdv = np.dot(A.T,dv)


        du = (w2 - sig2*dx)/sig1

        dlamu1 = (lam_u1/fu1)*(-dx+du) - lam_u1 - (1.0/tau)*1.0/fu1
        dlamu2 = (lam_u2/fu2)*(dx+du) - lam_u2 - 1.0/tau*1.0/fu2

        # make sure that the step is feasible: keeps lam_u1,lam_u2 > 0, fu1,fu2 < 0
        s = np.min(np.hstack([1.0, -lam_u1[dlamu1 < 0]/dlamu1[dlamu1 < 0], -lam_u2[dlamu2 < 0]/dlamu2[dlamu2 < 0]]))
        s = (0.99)*np.min(np.hstack([s, -fu1[(dx-du) > 0] / (dx[(dx-du) > 0] - du[(dx-du) > 0]) , -fu2[(-dx-du) > 0] / (-dx[(-dx-du) > 0] - du[(-dx-du) > 0]) ] ))

        # backtracking line search
        suffdec = 0
        backiter = 0
        while not(suffdec):
            xp = x + s*dx
            up = u + s*du
            vp = v + s*dv
            Atvp = Atv + s*Atdv
            lamu1p = lam_u1 + s*dlamu1
            lamu2p = lam_u2 + s*dlamu2
            fu1p = xp - up
            fu2p = -xp - up

            rdp = gradf0 + np.hstack([lamu1p-lamu2p, -lamu1p-lamu2p]) + np.hstack([Atvp, np.zeros(N)])
            rcp = np.hstack([-lamu1p*fu1p, -lamu2p*fu2p]) - (1.0/tau)
            rpp = r_pri + s*Adx
            suffdec = (np.linalg.norm(np.hstack([rdp, rcp, rpp])) <= (1.0-alpha*s)*resnorm)
            s = beta*s
            backiter = backiter + 1
            if backiter > 32:
                if verbose:
                    print('Stuck backtracking, returning last iterate.')
                xp = x.copy()
                return xp

        # next iteration
        x = xp.copy()
        u = up.copy()
        v = vp.copy()
        Atv = Atvp.copy()
        lam_u1 = lamu1p.copy()
        lam_u2 = lamu2p.copy()
        fu1 = fu1p.copy()
        fu2 = fu2p.copy()

        # surrogate duality gap
        sdg = -(np.sum(fu1*lam_u1) + np.sum(fu2*lam_u2))
        tau = mu*2.0*N/sdg
        r_pri = rpp.copy()
        r_cent = np.hstack([-lam_u1*fu1, -lam_u2*fu2]) - (1.0/tau)
        r_dual = gradf0 + np.hstack([lam_u1-lam_u2, -lam_u1-lam_u2]) + np.hstack([Atv, np.zeros(N)])
        resnorm = np.linalg.norm(np.hstack([r_dual, r_cent, r_pri]))

        done = (sdg < pdtol) | (pditer >= pdmaxiter)
        if verbose:
            print("Iteration = " + str(pditer) + ", tau = " + str(tau) + ", Primal = " + str(sum(u)) + ", PDGap = " + str(sdg) \
                + ", Dual res = " + str(np.linalg.norm(r_dual)) + ", Primal res = " + str(np.linalg.norm(r_pri)) )

            print("CG Res = " + str(cgres) + "CG Iter" + str(cgiter) )

    return xp
def get_bp_denoise(A, b, epsilon, x0 = None, lbtol = 1e-3, mu = 10, cgtol = 1e-8, cgmaxiter = 200, verbose = False, use_CG = False):
    """
    Solving the basis pursuit de-noising problem.

    :param numpy-matrix A:
        The matrix.
    :param numpy-array b:
        The right hand side column vector.
    :param double epsilon:
        The noise.
    :param numpy-array x0:
        Initial solution  if not provided the least norm solution is used.

    """
    newtontol = lbtol
    newtonmaxiter = 50

    b = b.flatten()

    # starting point --- make sure that it is feasible
    if not(x0 is None):
        if (np.linalg.norm(np.dot(A,x0) - b) > epsilon):
            if verbose:
                print('Starting point infeasible  using x0 = At*inv(AAt)*y.')
            if use_CG:
                w, cgres, cgiter =  __CG_solve(np.dot(A,A.T),b,cgmaxiter,cgtol)
            else:
                w = np.linalg.solve(np.dot(A,A.T),b).flatten()
                cgres = np.linalg.norm(np.dot(np.dot(A,A.T), w).flatten() - b.flatten()) / np.linalg.norm(b)
                cgiter = -1
            if cgres > .5:
              if verbose:
                    print("cgres = " + str(cgres) )
                    print('A*At is ill-conditioned: cannot find starting point' )
              xp = x0.copy()
              return xp
            x0 = np.dot(A.T, w)
    else:
        if verbose:
            print('No x0. Using x0 = At*inv(AAt)*y.')
        if use_CG:
            w, cgres, cgiter =  __CG_solve(np.dot(A,A.T),b,cgmaxiter,cgtol)
        else:
            w = np.linalg.solve(np.dot(A,A.T),b).flatten()
            cgres = np.linalg.norm(np.dot(np.dot(A,A.T), w).flatten() - b.flatten()) / np.linalg.norm(b)
            cgiter = -1
        if cgres > .5:
              if verbose:
                    print("cgres = " + str(cgres) )
                    print('A*At is ill-conditioned: cannot find starting point' )
        x0 = np.dot(A.T, w)

    x = x0.copy()
    r = np.reshape(np.dot(A, x), len(b)) - b
    N = len(x0)
    u = (0.95)*np.abs(x0) + (0.10)*np.max(np.abs(x0))     #arbitrary u starting point?
    if verbose:
        print('Original l1 norm = ' + str(np.sum(np.abs(x0))) + ', original functional = ' + str(np.sum(u)) )
    tau = np.max([(2.0*N+1)/np.sum(np.abs(x0)), 1.0])

    lbiter = int(np.ceil((np.log(2.0*N+1) - np.log(lbtol) - np.log(tau)) / np.log(mu)))
    if verbose:
        print('Number of log barrier iterations = ' + str(lbiter) )
    totaliter = 0
    for ii in range(lbiter+1):
      xp, up, ntiter =  __l1qc_newton(x, u, A, b, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter, verbose, use_CG)
      totaliter += ntiter
      if verbose:
          print('Log barrier iter = ' + str(ii) + ', l1 = ' + str(np.sum(np.abs(xp))) + ', functional = ' + str(np.sum(up)) + \
          ', tau = ' + str(tau) + ', total newton iter = ' + str(totaliter))

      x = xp.copy()
      u = up.copy()
      tau *= mu

    return xp

def __l1qc_newton(x0, u0, A, b, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter, verbose, use_CG):
    # line search parameters
    alpha = 0.01
    beta = 0.5
    AtA = np.dot(A.T,A)
    x = x0.flatten()
    u = u0.flatten()
    r = np.dot(A, x).flatten() - b.flatten()
    fu1 = x - u
    fu2 = -x - u
    fe = 0.5*(np.asscalar(np.dot(r.T,r)) - epsilon**2)
    f = np.sum(u) - (1.0/tau) * (np.sum(np.log(-fu1)) + np.sum(np.log(-fu2)) + np.log(-fe))

    niter = 0
    done = 0
    while (not(done)):

      atr = np.dot(A.T, r)

      ntgz = 1.0/fu1 - 1.0/fu2 + 1.0/fe * atr
      ntgu = -tau - 1.0/fu1 - 1.0/fu2
      gradf = - (1.0/tau) * np.hstack([ntgz, ntgu])

      sig11 = 1.0/fu1**2 + 1.0/fu2**2
      sig12 = -1.0/fu1**2 + 1.0/fu2**2
      sigx = sig11 - sig12**2/sig11

      w1p = ntgz - sig12/sig11 *ntgu

      H11p = np.diag(sigx.reshape(len(sigx))) - (1.0/fe) * AtA + (1.0/fe)**2 * np.outer(atr,atr)
      if use_CG:
          dx, cgres, cgiter =  __CG_solve(H11p, w1p, cgmaxiter, cgtol)
      else:
          dx = np.linalg.solve(H11p, w1p).flatten()
          cgres = np.linalg.norm(np.dot(H11p, dx).flatten() - w1p.flatten()) / np.linalg.norm(w1p)
          cgiter = -1
      if (cgres > 0.5):
          if verbose:
              print("cgres = " + str(cgres) )
              print('Cannot solve system.  Returning previous iterate.' )
          xp = x.flatten()
          up = u.flatten()
          return xp, up, 0
      Adx = np.dot(A,dx).flatten()


      du = (1.0/sig11) * ntgu - (sig12/sig11)*dx

#     minimum step size that stays in the interior
      aqe = np.dot(Adx.T, Adx)
      bqe = 2.0*np.dot(r.T, Adx)
      cqe = np.asscalar(np.dot(r.T,r)) - epsilon**2

      smax = np.min(np.hstack([ 1.0,np.min(np.hstack([-fu1[(dx-du) > 0] / (dx[(dx-du) > 0] - du[(dx-du) > 0]),\
        -fu2[(-dx-du) > 0] / (-dx[(-dx-du) > 0] - du[(-dx-du) > 0]), \
        np.reshape((-bqe + np.sqrt(bqe**2 - 4 * aqe * cqe)) / (2.0*aqe), (1,)) ] ))]))
      s = (0.99) * smax

      # backtracking line search
      suffdec = 0
      backiter = 0
      while not(suffdec):
        xp = x + s*dx
        up = u + s*du
        rp = r + s*Adx

        fu1p = xp - up
        fu2p = -xp - up

        fep = 0.5 * (np.linalg.norm(rp)**2 - epsilon**2)
        fp = np.sum(up) - (1.0/tau) * (np.sum(np.log(-fu1p)) + np.sum(np.log(-fu2p)) + np.log(-fep))

        flin = f + alpha * s * (np.dot(gradf.T, np.hstack([dx, du])))

        suffdec = (fp <= flin)
        s = beta * s
        backiter = backiter + 1
        if (backiter > 32):
          if verbose:
              print('Stuck on backtracking line search, returning previous iterate.')
          xp = x.copy()
          up = u.copy()
          return xp,up,niter

      # set up for next iteration
      x = xp.copy()
      u = up.copy()
      r = rp.copy()
      fu1 = fu1p.copy()
      fu2 = fu2p.copy()
      fe = fep.copy()
      f = fp.copy()

      lambda2 = -(np.dot(gradf, np.hstack([dx, du])))
      stepsize = s*np.linalg.norm(np.hstack([dx, du]))
      niter = niter + 1
      done = (lambda2/2 < newtontol) | (niter >= newtonmaxiter)
      if verbose:
          print('Newton iter = ' + str(niter) + ', Functional = ' + str(f) + ', Newton decrement = ' + str(lambda2/2) + ', Stepsize = ' + str(stepsize))
          print('                CG Res = ' + str(cgres) + ', CG Iter = ' + str(cgiter))
    return xp, up, niter