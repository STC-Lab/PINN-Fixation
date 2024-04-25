import matplotlib.pyplot as plt


# visualize the dataset and the results

#Draw the data points and physics points
def plotdataphysics(x_data,t_data,x_physics,t_physics):
    # visualize collocation points for 2D input space (x, t)
    plt.figure()
    plt.scatter(x_data.detach().numpy(), t_data.detach().numpy(),s=4., c='blue', marker='o', label='Data points')
    plt.scatter(x_physics.detach().numpy(), t_physics.detach().numpy(),s=4., c='red', marker='o', label='Physics points')
    plt.title('Samples of the PDE solution y(x,t) for training')
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend('Data points and physics points')
    plt.show()
    plt.show(block=True)


#Draw a 3D graph to visulize u(x,t)
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