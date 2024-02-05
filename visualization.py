import matplotlib.pyplot as plt


def plot3D(X, T, y):
    X = X.detach().numpy()
    T = T.detach().numpy()
    y = y.detach().numpy()

    #     2D
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    cm = ax1.contourf(T, X, y, 20,cmap="viridis")
    fig.colorbar(cm, ax=ax1) # Add a colorbar to a plot
    ax1.set_title('u(x,t)')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_aspect('equal')
        #     3D
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(T, X, y,cmap="viridis")
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('u(x,t)')
    fig.tight_layout()
    plt.show()

def plotdata(X_train_Nu,T_train_Nu):
    # visualize collocation points for 2D input space (x, t)
    plt.figure()
    plt.scatter(X_train_Nu.detach().numpy(), T_train_Nu.detach().numpy(),s=4., c='blue', marker='o', label='CP')
    plt.title('Samples of the PDE solution y(x,t) for training')
    plt.xlim(1., 1.)
    plt.ylim(-1., 1.)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend(loc='upper right')
    plt.show()
    plt.show(block=True)
