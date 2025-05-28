#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：Lotka-Volterra捕食者-猎物模型 - 学生代码模板

学生姓名：[请填写您的姓名]
学号：[请填写您的学号]
完成日期：[请填写完成日期]
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def lotka_volterra_system(state: np.ndarray, t: float, alpha: float, beta: float,
                        gamma: float, delta: float) -> np.ndarray:
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = gamma * x * y - delta * y
    return np.array([dxdt, dydt])


def euler_method(f, y0: np.ndarray, t_span: Tuple[float, float],
                 dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(len(t) - 1):
        y[i + 1] = y[i] + dt * f(y[i], t[i], *args)

    return t, y


def improved_euler_method(f, y0: np.ndarray, t_span: Tuple[float, float],
                         dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(len(t) - 1):
        k1 = dt * f(y[i], t[i], *args)
        k2 = dt * f(y[i] + k1, t[i] + dt, *args)
        y[i + 1] = y[i] + 0.5 * (k1 + k2)

    return t, y


def runge_kutta_4(f, y0: np.ndarray, t_span: Tuple[float, float],
                  dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(len(t) - 1):
        k1 = dt * f(y[i], t[i], *args)
        k2 = dt * f(y[i] + 0.5 * k1, t[i] + 0.5 * dt, *args)
        k3 = dt * f(y[i] + 0.5 * k2, t[i] + 0.5 * dt, *args)
        k4 = dt * f(y[i] + k3, t[i] + dt, *args)
        y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t, y


def solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt):
    y0_vec = np.array([x0, y0])
    t, sol = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    x, y = sol[:, 0], sol[:, 1]
    return t, x, y


def compare_methods(alpha, beta, gamma, delta, x0, y0, t_span, dt):
    y0_vec = np.array([x0, y0])
    t_e, sol_e = euler_method(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    t_ie, sol_ie = improved_euler_method(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)
    t_rk, sol_rk = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, alpha, beta, gamma, delta)

    return {
        'euler': {'t': t_e, 'x': sol_e[:, 0], 'y': sol_e[:, 1]},
        'improved_euler': {'t': t_ie, 'x': sol_ie[:, 0], 'y': sol_ie[:, 1]},
        'rk4': {'t': t_rk, 'x': sol_rk[:, 0], 'y': sol_rk[:, 1]}
    }


def plot_population_dynamics(t, x, y, title="Lotka-Volterra种群动力学"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t, x, label='猎物 (x)')
    plt.plot(t, y, label='捕食者 (y)')
    plt.xlabel("时间")
    plt.ylabel("种群数量")
    plt.title("时间序列图")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, y)
    plt.xlabel("猎物数量 x")
    plt.ylabel("捕食者数量 y")
    plt.title("相空间轨迹图")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_method_comparison(results):
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))

    methods = ['euler', 'improved_euler', 'rk4']
    titles = ['欧拉法', '改进欧拉法', 'RK4']

    for i, method in enumerate(methods):
        t = results[method]['t']
        x = results[method]['x']
        y = results[method]['y']

        axs[0, i].plot(t, x, label='x (猎物)')
        axs[0, i].plot(t, y, label='y (捕食者)')
        axs[0, i].set_title(f"{titles[i]} 时间序列")
        axs[0, i].legend()

        axs[1, i].plot(x, y)
        axs[1, i].set_title(f"{titles[i]} 相空间")

    plt.tight_layout()
    plt.show()


def analyze_parameters():
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    t_span = (0, 30)
    dt = 0.01

    initial_conditions = [
        (1.0, 2.0),
        (2.0, 1.0),
        (3.0, 3.0),
        (1.5, 2.5)
    ]

    plt.figure(figsize=(10, 8))
    for x0, y0 in initial_conditions:
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        plt.plot(x, y, label=f"x0={x0}, y0={y0}")

    plt.title("不同初始条件的相空间轨迹")
    plt.xlabel("猎物数量 x")
    plt.ylabel("捕食者数量 y")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 守恒量验证
    def conserved_quantity(x, y):
        return delta * np.log(x) - gamma * x + beta * y - alpha * np.log(y)

    t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, 2.0, 2.0, t_span, dt)
    H = conserved_quantity(x, y)

    plt.plot(t, H)
    plt.title("守恒量随时间的变化")
    plt.xlabel("时间")
    plt.ylabel("守恒量 H")
    plt.grid(True)
    plt.show()


def main():
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    x0, y0 = 2.0, 2.0
    t_span = (0, 30)
    dt = 0.01

    print("=== Lotka-Volterra捕食者-猎物模型分析 ===")
    print(f"参数: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    print(f"初始条件: x0={x0}, y0={y0}")
    print(f"时间范围: {t_span}, 步长: {dt}")

    try:
        print("\n1. 使用4阶龙格-库塔法求解...")
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        plot_population_dynamics(t, x, y)

        print("\n2. 比较不同数值方法...")
        results = compare_methods(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        plot_method_comparison(results)

        print("\n3. 分析参数影响...")
        analyze_parameters()

        print("\n4. 数值结果统计:")
        print(f"最大猎物数量: {np.max(x):.2f}")
        print(f"最小捕食者数量: {np.min(y):.2f}")
        print(f"平均猎物数量: {np.mean(x):.2f}")
        print(f"平均捕食者数量: {np.mean(y):.2f}")

    except NotImplementedError as e:
        print(f"\n错误: {e}")
        print("请完成相应函数的实现后再运行主程序。")


if __name__ == "__main__":
    main()
