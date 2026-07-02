import numpy as np
import matplotlib.pyplot as plt

def lin_func(
        x: np.ndarray,
        p: np.ndarray = np.array([[5,-10]]),
) -> np.ndarray:
    """Simple linear function to be learned"""
    if x.ndim == 1:
        x = np.reshape(x, (1, -1))
    a, b=p[:,0].reshape(-1,1 ), p[:,1].reshape(-1,1)
    return a*x + b

def mse_loss(
        y_hat: np.ndarray,
        y: np.ndarray,
) -> np.ndarray:
    """Mean Squared Error Loss Function"""
    return np.mean((y-y_hat)**2, axis=-1)

def linear_regression(
        epochs: int=10000,
        verbose:bool=True,
        dx: float=1e-5,
        lr: float=1e-3,
) -> None:
    p = np.array([[1, 0]])
    x = np.linspace(-1, 1, 100)
    y = lin_func(x)[0]
    if verbose:
        plt.plot(x, y, '.', color='k', label="data")
        plt.plot(x, lin_func(x, p)[0], '--', color='b', linewidth=5, label="Initial fit")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.tight_layout()
        plt.show()

    for i in range(epochs):
        dp = dx*np.eye(p.shape[-1]) + p
        curr_preds = lin_func(x, p)
        new_preds = lin_func(x, dp)
        grads = (mse_loss(new_preds, y) - mse_loss(curr_preds, y))/dx
        p = p - lr * grads
        if verbose:
            print(f"epoch: {i}, p: {p}, loss: {mse_loss(lin_func(x, p), y)}")

    if verbose:
        plt.plot(x, lin_func(x, p)[0], '--', color='b', label="Trained fit")
        plt.plot(x, y, '.', color='r', label="Data")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.tight_layout()
        plt.show()




def main() -> None:
    linear_regression()

if __name__ == "__main__":
    main()