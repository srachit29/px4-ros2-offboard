"""
mpc.py - linear Model Predictive Control for one horizontal axis of a quadrotor.

Model (identified from flight logs with tools/fit_lag.py):
    p' = v
    v' = a
    a' = (u - a) / tau          u = commanded acceleration, tau ~ 0.2 s
plus a pure input delay of ~0.08 s, handled by predicting the state forward
through the commands that were already sent but have not acted yet.

At every control step the MPC solves
    minimise   sum_{k=1..N}  q_p (p_k - p_ref_k)^2 + q_v (v_k - v_ref_k)^2
             + sum_{k=0..N-1} r (u_k - a_ref_k)^2 + r_du (u_k - u_{k-1})^2
    subject to |u_k| <= u_max
and applies only the first command u_0 (receding horizon).

Because the model is linear and time-invariant, all prediction matrices are
built once. The box constraint is handled with a projected gradient method,
which needs only numpy and runs in well under a millisecond for N = 30.
"""
import numpy as np


class AxisMPC:
    def __init__(self, dt=0.05, horizon=30, tau=0.2, q_p=20.0, q_v=2.0, r=1.0,
                 r_du=2.0, u_max=3.0):
        self.dt, self.N, self.tau, self.u_max = dt, horizon, tau, u_max
        N = horizon

        # discrete model x_{k+1} = A x_k + B u_k  (exact discretisation of the lag)
        e = np.exp(-dt / tau)
        A = np.array([[1.0, dt, tau * dt - tau ** 2 * (1 - e)],
                      [0.0, 1.0, tau * (1 - e)],
                      [0.0, 0.0, e]])
        B = np.array([dt ** 2 / 2 - (tau * dt - tau ** 2 * (1 - e)),
                      dt - tau * (1 - e),
                      1 - e]).reshape(3, 1)
        self.A, self.B = A, B

        # stacked prediction  X = Phi x0 + Gam U   (X = [x_1 .. x_N])
        Phi = np.zeros((3 * N, 3))
        Gam = np.zeros((3 * N, N))
        Ak = np.eye(3)
        for k in range(N):
            Ak = A @ Ak
            Phi[3 * k:3 * k + 3] = Ak
            for j in range(k + 1):
                Gam[3 * k:3 * k + 3, j] = (np.linalg.matrix_power(A, k - j) @ B).ravel()
        self.Phi, self.Gam = Phi, Gam

        Q = np.kron(np.eye(N), np.diag([q_p, q_v, 0.0]))
        D = np.eye(N) - np.eye(N, k=-1)                 # u_k - u_{k-1}
        self.Q, self.D = Q, D
        self.r, self.r_du = r, r_du
        H = Gam.T @ Q @ Gam + r * np.eye(N) + r_du * D.T @ D
        self.H = H
        self.step = 1.0 / np.linalg.eigvalsh(H).max()   # safe gradient step
        self.H_inv = np.linalg.inv(H)
        self.u_prev_plan = np.zeros(N)

    def solve(self, x0, p_ref, v_ref, a_ref, u_last):
        """x0 = [p, v, a] now; p_ref, v_ref, a_ref = arrays of length N (k = 1..N
        for p, v and k = 0..N-1 for a); u_last = command applied in the last step."""
        N = self.N
        x0 = np.asarray(x0, dtype=float)
        p_ref = np.asarray(p_ref, dtype=float)
        v_ref = np.asarray(v_ref, dtype=float)
        a_ref = np.asarray(a_ref, dtype=float)
        X_ref = np.zeros(3 * N)
        X_ref[0::3], X_ref[1::3] = p_ref, v_ref
        d0 = np.zeros(N)
        d0[0] = u_last
        f = (self.Gam.T @ self.Q @ (self.Phi @ x0 - X_ref)
             - self.r * a_ref - self.r_du * self.D.T @ d0)
        U = -self.H_inv @ f                              # unconstrained optimum
        if np.max(np.abs(U)) > self.u_max:               # constraints active
            U = np.clip(np.concatenate([self.u_prev_plan[1:], self.u_prev_plan[-1:]]),
                        -self.u_max, self.u_max)
            for _ in range(60):
                U = np.clip(U - self.step * (self.H @ U + f), -self.u_max, self.u_max)
        self.u_prev_plan = U
        return float(U[0]), U


def predict_through_delay(x, pending, dt_ctrl, tau):
    """Advance state x = [p, v, a] through commands already sent but not yet acting."""
    p, v, a = x
    for u in pending:
        a += (u - a) * dt_ctrl / tau
        v += a * dt_ctrl
        p += v * dt_ctrl
    return np.array([p, v, a])
