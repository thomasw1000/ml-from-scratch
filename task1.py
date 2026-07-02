import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def lin_func(
        x: np.ndarray,
        p: np.ndarray = np.array([[5,-10]]),
) -> np.ndarray:
    """Simple linear function to be learned"""
    if x.ndim == 1:
        x = np.reshape(x, (1, -1))
    a, b=p[:,0].reshape(-1,1), p[:,1].reshape(-1,1)
    return a*x + b

def mse_loss(
        y_hat: np.ndarray,
        y: np.ndarray,
) -> np.ndarray:
    """Mean Squared Error Loss Function"""
    return np.mean((y-y_hat)**2, axis=-1)

def sigmoid(
        x: np.ndarray,
) -> np.ndarray:
    return 1/(1+np.exp(-x))

def generate_labels(
        x: np.ndarray,
        p : np.ndarray = np.array([[-2, 1]])
) -> np.ndarray:
    sig = sigmoid(lin_func(x, p))
    noise = np.random.normal(0, 0.1, x.shape)
    return (sig>0.5+noise).astype(int)

def binary_cross_entropy(
        y: np.ndarray,
        y_hat: np.ndarray,
) -> np.ndarray:
    return -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat), axis=-1)

def plot_classes(
        x: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
) -> None:
    plt.plot(x, y, '.', color='k', label="Data", markersize=5, alpha=0.5)
    plt.plot(x, y_pred, '.', color='b', label="Predicted", markersize=5, alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('Class')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_fit(
        x: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
) -> None:
    plt.plot(x, y, '.', color='k', label="Data", markersize=5)
    plt.plot(x, y_pred, '--', color='b', linewidth=1, label="Predicted")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()


def linear_regression(
        epochs: int=10000,
        verbose:bool=True,
        dx: float=1e-5,
        lr: float=1e-3,
) -> np.ndarray:
    p = np.array([[1, 0]])
    x = np.linspace(-1, 1, 100)
    y = lin_func(x)[0]
    if verbose:
        plot_fit(x, y, lin_func(x, p)[0])

    for i in range(epochs):
        dp = dx*np.eye(p.shape[-1]) + p
        curr_preds = lin_func(x, p)
        new_preds = lin_func(x, dp)
        grads = (mse_loss(new_preds, y) - mse_loss(curr_preds, y))/dx
        p = p - lr * grads
        if verbose:
            print(f"epoch: {i}, p: {p}, loss: {mse_loss(lin_func(x, p), y)}")

    if verbose:
        plot_fit(x, y, lin_func(x, p)[0])

    return p

def logistic_regression(
        epochs: int=10000,
        verbose: bool=True,
        dx: float=1e-3,
        lr: float=1e-3
) -> np.ndarray:
    p = np.array([[1, 0]])
    x = np.linspace(-2, 2, 200)
    labels = generate_labels(x, p).reshape(-1)
    if verbose:
        plot_classes(x, labels, (sigmoid(lin_func(x, p))).astype(int).reshape(-1))
    for i in range(epochs):
        dp = dx*np.eye(p.shape[-1]) + p
        curr_preds = sigmoid(lin_func(x, p))
        new_preds = sigmoid(lin_func(x, dp))
        grads = (binary_cross_entropy(labels, new_preds)-binary_cross_entropy(labels, curr_preds))/dx
        p = p - lr*grads
        if verbose:
            print(f"epoch: {i}, p: {p}, loss: {binary_cross_entropy(labels, sigmoid(lin_func(x, p)))}")
    if verbose:
        plot_classes(x, labels, (sigmoid(lin_func(x, p))).astype(int).reshape(-1))
    return p



def main() -> None:
    lin_p = linear_regression()
    print(f"Best fit linear regression params: {lin_p}")
    log_p = logistic_regression()
    print(f"Best fit logistic regression params: {log_p}")

if __name__ == "__main__":
    main()