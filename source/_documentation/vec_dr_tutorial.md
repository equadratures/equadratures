Vector-valued Dimension Reduction
====================
In some applications, we may be interested in the approximation of vector-valued objectives. This allows us to study the dimension reducing subspaces of multiple scalar objectives, such as the lift and drag of an airfoil simultaneously, or may be interested in the discretized pressure profile over this airfoil. With an approach that extends the technique of [active subspaces](https://effective-quadratures.github.io/Effective-Quadratures/_documentation/tutorial_11.html), we can find subspaces within which the majority of the variation vector-valued function will be contained. In other words, for a function $\mathbf{f}(\mathbf{x}): \mathbb{R}^d \rightarrow \mathbb{R}^n$, we want to find a ridge approximation


.. math::

        \mathbf{f}(\mathbf{x}) \approx \mathbf{g}( \mathbf{U}^T \mathbf{x}),


where $\mathbf{U} \in \mathbb{R}^{k \times d}$ with $k \ll d$. In EQ, we base our approach on Zahm et al. [1] and study the vector gradient covariance matrix. Define the Jacobian matrix as

.. math::

        \mathbf{J}(\mathbf{x}) = \left[\frac{\partial f_1}{\partial \mathbf{x}} , \frac{\partial f_2}{\partial \mathbf{x}} ,..., \frac{\partial f_n}{\partial \mathbf{x}} \right],


where $f_i$ is the $i$-th output component of $\mathbf{f}$. This matrix is constructed by stacking together the gradient vectors $\frac{\partial f_i}{\partial \mathbf{x}}$. How are the gradients evaluated? Assuming that automatic differentiation or adjoints are not available, we can adopt an approach similar to that detailed in the [active subspaces tutorial](https://effective-quadratures.github.io/Effective-Quadratures/_documentation/tutorial_11.html), and assume that each $f_i$ has already been approximated as a polynomial series,

.. math::

        f_i(\mathbf{x}) \approx \sum_{j=1}^P a_j \phi_j(\mathbf{x}),


from which gradients are easily evaluated. Then, we can form the gradient covariance matrix $\mathbf{H}$ as

.. math::

        \mathbf{H} = \int \mathbf{J}(\mathbf{x}) \mathbf{R} \mathbf{J}(\mathbf{x})^T \rho(\mathbf{x}) d\mathbf{x}.


In this expression, the positive semi-definite matrix $\mathbf{R}$ represents the weighting of objectives. We can place more weight on some objectives than others, analogous to a weighted objective function derived from multiple objectives in an optimization problem. 

We can then compute the eigendecomposition of $\mathbf{H}$ as in the scalar output case, and retain the eigenvectors corresponding to the largest eigenvalues as the columns of the dimension reducing matrix $\mathbf{U}$. A Monte-Carlo based strategy for vector-valued dimension reduction using EQ is then

1. For each component $f_i(\mathbf{x})$ of the function $\mathbf{f}$, we fit a polynomial surrogate, $f_i(\mathbf{x}) \approx p_i(\mathbf{x}) = \sum_{j=1}^P a_j \phi_j(\mathbf{x}),$
2. Sample each polynomial at $M$ points drawn from the input distribution $\rho(\mathbf{x})$ and evaluate the gradient at each point.
3. Form the Jacobian matrix using the polynomial surrogates at each input point, $\mathbf{J}(\mathbf{x}_i) = \left[\frac{\partial p_1}{\partial \mathbf{x}}(\mathbf{x}_i) , \frac{\partial p_2}{\partial \mathbf{x}} (\mathbf{x}_i),..., \frac{\partial p_n}{\partial \mathbf{x}}  (\mathbf{x}_i)\right]$ for $i = 1,...,M$.
4. Evaluate the vector gradient covariance matrix with a prescribed $\mathbf{R}$, 

.. math::

        \mathbf{H} = \frac{1}{M}\sum_{i=1}^M \mathbf{J}(\mathbf{x}_i) \mathbf{R} \mathbf{J}(\mathbf{x}_i)^T


5. Compute the eigendecomposition of $\mathbf{H}$ and partition the eigenvectors with large eigenvalues to obtain $\mathbf{U}$:

.. math::

        \mathbf{H} = [\mathbf{U}\quad \mathbf{V}] \begin{bmatrix} \mathbf{\Lambda}_1 & \\ & \mathbf{\Lambda}_2 \end{bmatrix} [\mathbf{U} \quad \mathbf{V}]^T,


**Code Implementation**

We consider a simple analytical example to demonstrate the use of vector-valued dimension reduction routines. Consider the function $\mathbf{f}: [-1,1]^5 \rightarrow \mathbb{R}^2$, with $\mathbf{f}(\mathbf{x}) = [f_0(\mathbf{x}), f_1(\mathbf{x})]^T$ where:

.. math::

        f_0(\mathbf{x}) = \sin(\pi \mathbf{w}_0^T \mathbf{x})

.. math::

        f_1(\mathbf{x}) = \exp(\mathbf{w}_1^T \mathbf{x})


.. code::

        import numpy as np
        from equadratures import *
        d = 5
        n = 2

        def f_0(x,w):
                return np.sin(np.dot(x, w) * np.pi)

        def f_1(x,w):
                return np.exp(np.dot(x,w))

We define $\mathbf{W} = [\mathbf{w}_0, \mathbf{w}_1]$ as random orthogonal vectors...

.. code::

        rand_norm = np.random.randn(d,n)
        W,_ = np.linalg.qr(rand_norm)

and use 1000 samples to construct seventh-degree polynomials for the component functions:

.. code::

        poly_list = []
        N = 1000
        p = 7
        myBasis = Basis('Total order', [p for _ in range(d)])
        params = [Parameter(order=p, distribution='uniform', lower=-1, upper=1) for _ in range(d)]
        X_train = np.random.uniform(-1, 1, size=(N, d))
        Y_train = np.zeros((N, n))
        Y_train[:,0] = np.apply_along_axis(f_0, 1, X_train, W[:,0])
        Y_train[:,1] = np.apply_along_axis(f_1, 1, X_train, W[:,1])
        for i in range(n):
        poly_list.append(Polyreg(params, myBasis, training_inputs=X_train, training_outputs=Y_train[:,i],
                                no_of_quad_points=0))

Now we can call the vector_AS method in the dr class to compute the Jacobian and find the DR space $\mathbf{U}$.

.. code::

        my_dr = dr(training_input=X_train)
        R = np.eye(n)
        [eigs, U] = my_dr.vector_AS(poly_list, R=R)
        U = np.real(U)

To check whether our answer is sensible, we can check the subspace distance between $\mathbf{U}$ and the space spanned by the true dimension reducing subspace, $\text{span}(\mathbf{w}_0, \mathbf{w}_1)$:

.. math::

        \text{dist}(\mathbf{U}, \mathbf{W}) = ||\mathbf{U}\mathbf{U}^T - \mathbf{W}\mathbf{W}^T||_2


where the 2-norm picks out the largest singular value of the argument.

.. code::

        def subspace_dist(A, B):
                return np.linalg.norm(np.dot(A, A.T) - np.dot(B, B.T), ord=2)
        
        print(subspace_dist(W,np.array(U[:,:n])))

We can also verify that the variation of both functions outside of $\mathbf{U}$ is small. Here, for each of 10 sets of active coordinates, we sample 50 points in the inactive subspace ($\mathbf{V}$) using the "hit and run" algorithm from Python Active-subspaces Utility Library [2]. 

.. code::

        from active_subspaces.domains import hit_and_run_z
        def harz(W1,W2,y,N):
                U = np.hstack([W1,W2])
                Z = hit_and_run_z(N, y, W1, W2)
                yz = np.vstack([np.repeat(y[:,np.newaxis], N, axis=1), Z.T])
                return np.dot(U, yz).T

        N_inactive = 50
        plot_coords = np.zeros((N_inactive, 2, 10))
        for t in range(10):
                rand_active_coords = np.random.uniform(-1,1,2)

                new_X = harz(U[:,:n], U[:,n:], rand_active_coords, N_inactive)

                new_f0 = np.apply_along_axis(f_0, 1, new_X, W[:,0])
                new_f1 = np.apply_along_axis(f_1, 1, new_X, W[:,1])

                plot_coords[:,0,t] = new_f0
                plot_coords[:,1,t] = new_f1

        plt.figure()
        for t in range(10):
                plt.scatter(plot_coords[:, 0, t], plot_coords[:, 1, t], s=3)

.. figure:: inactive.png
        :scale: 100 %

It can be seen that at each active coordinate, the variation in the inactive subspace is small.

**References**

.. [1] Zahm, O.,  Constantine, P., Prieur, C. and Marzouk, Y., Gradient-based dimension reduction of multi-variate vector-valued functions, arXiv:1801.07922 [math], (2018), http://arxiv.org/abs/1801.07922 (accessed
2018-11-12). arXiv: 1801.07922.

.. [2] Constantine, P., Howard, R., Glaws, A., Grey, Z., Diaz, P., & Fletcher, L. (2016, September 29). Python Active-subspaces Utility Library. Zenodo. http://doi.org/10.5281/zenodo.158941

