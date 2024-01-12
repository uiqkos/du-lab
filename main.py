import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from IPython.display import clear_output
from torch import Tensor
from torch.autograd import Variable
from torch.nn import functional as F

from modules import NeuralODE, LinearODEF

sns.color_palette("bright")
use_cuda = torch.cuda.is_available()


def plot_trajectories(obs=None, times=None, trajs=None, save=None, figsize=(16, 8)):
    plt.figure(figsize=figsize)
    max_x = max_y = min_x = min_y = 0

    if obs is not None:
        if times is None:
            times = [None] * len(obs)

        for o, t in zip(obs, times):
            o, t = o.detach().cpu().numpy(), t.detach().cpu().numpy()

            # Рисуем точки для правильных значений
            for b_i in range(o.shape[1]):
                plt.scatter(o[:, b_i, 0], o[:, b_i, 1], c=t[:, b_i, 0], cmap=cm.plasma)
                max_x = max(max_x, o[:, b_i, 0].max())
                max_y = max(max_y, o[:, b_i, 1].max())
                min_x = min(min_x, o[:, b_i, 0].min())
                min_y = min(min_y, o[:, b_i, 1].min())

    # Рисуем значения предсказания
    if trajs is not None:
        for z in trajs:
            z = z.detach().cpu().numpy()
            plt.plot(z[:, 0, 0], z[:, 0, 1], lw=1.5)

        # Сохраняем картинку
        if save is not None:
            if not os.path.exists(os.path.dirname(save)):
                os.makedirs(os.path.dirname(save))
            plt.savefig(save)

    # Цветовая шкала
    plt.colorbar()
    # Границы
    plt.xlim(min_x - 0.1, max_x + 0.1)
    plt.ylim(min_y - 0.1, max_y + 0.1)

    plt.show()


def experiment(ode_true, ode_trained, n_steps, name, plot_freq=10):
    # Генерируем данные
    z0 = Variable(torch.Tensor([[0.6, 0.3]]))

    t_max = 6.29 * 5
    n_points = 200

    index_np = np.arange(0, n_points, 1, dtype=np.int_)
    index_np = np.hstack([index_np[:, None]])
    times_np = np.linspace(0, t_max, num=n_points)
    times_np = np.hstack([times_np[:, None]])

    # Время
    times = torch.from_numpy(times_np[:, :, None]).to(z0)

    # Правдивые значения
    obs = ode_true(z0, times, return_whole_sequence=True).detach()

    # Добавляем шум
    obs = obs + torch.randn_like(obs) * 0.01

    # Параметры обучения
    min_delta_time = 1.0
    max_delta_time = 5.0
    max_points_num = 32

    optimizer = torch.optim.Adam(ode_trained.parameters(), lr=0.01)

    for i in range(n_steps):
        # Случайно выбираем интервал времени
        t0 = np.random.uniform(0, t_max - max_delta_time)
        t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)

        idx = sorted(np.random.permutation(index_np[(times_np > t0) & (times_np < t1)])[:max_points_num])

        obs_ = obs[idx]
        ts_ = times[idx]

        # Считаем предсказание
        z_ = ode_trained(obs_[0], ts_, return_whole_sequence=True)
        # Считаем ошибку
        loss = F.mse_loss(z_, obs_.detach())

        # Обновляем веса
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Выводим результаты
        if i % plot_freq == 0:
            z_p = ode_trained(z0, times, return_whole_sequence=True)

            plot_trajectories(obs=[obs], times=[times], trajs=[z_p], save=f"assets/imgs/{name}/{i}.png")
            clear_output(wait=True)


# Правдивая модель
ode_true = NeuralODE(LinearODEF(Tensor([[0, -0.2], [2., -0.5]])))

# Обучаемая модель
ode_trained = NeuralODE(LinearODEF(torch.randn(2, 2) / 2.))

# Проводим эксперимент с 1000 шагами
experiment(ode_true, ode_trained, 1000, "linear")
