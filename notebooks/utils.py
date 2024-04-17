from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def turn_into_two_vars(f, X, token1, token2):
    def f_new(t1_val, t2_val):
        if np.isnan(t1_val) or np.isinf(t1_val):
            t1_val = 1e10
        if np.isinf(t2_val) or np.isnan(t2_val):
            t2_val = 1e10
        X[token1] = t1_val
        X[token2] = t2_val
        return f(X)
    return f_new


@dataclass
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

def plot_2d(AMM: Tuple[Callable, Callable], X: np.array, token1: int, token2: int, bounds: Bounds, n=100, label=None, color="red"):
    X = X.copy() + 1e-9 # add small epsilon to get curves to touch each axis
    x0 = X[token1]
    y0 = X[token2]
    f, f_grad = AMM
    f_new = turn_into_two_vars(f, X, token1, token2)
    f_grad_new = turn_into_two_vars(f_grad, X, token1, token2)


    def hyperbola_ode(t, z):
        x, y = z
        grad = f_grad_new(x, y)
        if grad[1] == 0:
            grad[1] = 1e-9
        tangent = [1, -grad[0]/grad[1]]
        return tangent
    
    def stop_on_unstable(t, z):
        x, y = z
        grad = f_grad_new(x, y)
        # if np.isnan(grad).any() or np.abs(grad).max() > 1e9:
        #     print("stopping")
        #     return 0
        return 1
    
    stop_on_unstable.terminal = True
    
    t_span_forward = [x0, bounds.x_max]  # Forward in x from x0 to 5
    initial_state = [x0, y0]
    sol_forward = solve_ivp(hyperbola_ode, t_span_forward, initial_state, t_eval=np.linspace(x0, bounds.x_max, n), method='RK45', events=stop_on_unstable)

    # Solve ODE backward from the middle point
    t_span_backward = [x0, bounds.x_min]  # Backward in x from x0 to near zero
    sol_backward = solve_ivp(hyperbola_ode, t_span_backward, initial_state, t_eval=np.linspace(x0, bounds.x_min, n), method='RK45', events=stop_on_unstable)

    # Concatenate solutions: Reverse the backward solution and append the forward solution
    x_combined = np.concatenate((sol_backward.y[0][::-1], sol_forward.y[0][1:]))
    y_combined = np.concatenate((sol_backward.y[1][::-1], sol_forward.y[1][1:]))

    # Plotting the result
    # plt.figure(figsize=(8, 6))
    plt.xlim(0, bounds.x_max)
    plt.ylim(0, bounds.y_max)
    plt.plot(x_combined, y_combined, label=label, lw=1)
    plt.scatter(x0, y0, color=color, zorder=2)  # Mark the starting point
    plt.xlabel(f"Token {token1}")
    plt.ylabel(f"Token {token2}")
    plt.legend()
    # plt.show()
