import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5agg')
plt.ion()

A = np.ones((200, 200))
x = np.linspace(0, 1, 200)
y = np.ones(200)

fig = plt.figure()
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
im = ax0.imshow(A, vmin=0, vmax=1)
pl, = ax1.plot(x, y)

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return




while True:
    A = np.random.random((200, 200))
    y = np.random.random(200)
    im.set_data(A)
    pl.set_data(x, y)
    mypause(0.01)
    fig.canvas.draw()
plt.show(block=False)
