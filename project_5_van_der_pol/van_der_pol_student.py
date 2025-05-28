#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
van der Pol振荡器模拟程序
该程序模拟并可视化van der Pol振荡器的动力学行为，包括时间演化、相空间轨迹和极限环分析。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple, Callable, List

def van_der_pol_ode(t, state, mu=1.0, omega=1.0):
    """
    van der Pol振子的一阶微分方程组。
    
    参数:
        t: 时间变量(未直接使用，因ODE标准形式需要保留)
        state: 当前状态向量 [x, v]
        mu: 非线性阻尼系数
        omega: 自然频率
    
    返回:
        numpy数组: [dx/dt, dv/dt]
    """
    x, v = state  # 解包状态变量
    return np.array([v, mu*(1-x**2)*v - omega**2*x])  # van der Pol方程

def solve_ode(ode_func, initial_state, t_span, dt, **kwargs):
    """
    使用solve_ivp求解常微分方程组
    
    参数:
        ode_func: 微分方程函数
        initial_state: 初始状态
        t_span: 时间范围 (t_start, t_end)
        dt: 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        tuple: (时间数组, 状态数组)
    """
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)  # 生成时间点
    sol = solve_ivp(ode_func, t_span, initial_state, 
                   t_eval=t_eval, args=tuple(kwargs.values()), method='RK45')
    return sol.t, sol.y.T  # 返回时间和状态轨迹

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态变量的时间演化图
    
    参数:
        t: 时间数组
        states: 状态数组 (每列是一个状态变量)
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='Position x(t)')  # 位置x随时间变化
    plt.plot(t, states[:, 1], label='Velocity v(t)')  # 速度v随时间变化
    plt.xlabel('Time t')
    plt.ylabel('State Variables')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹图
    
    参数:
        states: 状态数组 [x, v]
        title: 图表标题
    """
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1])  # 相空间轨迹 (x vs v)
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')  # 保持纵横比一致
    plt.show()

def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    """
    分析极限环的特征（振幅和周期）
    
    参数:
        states: 状态数组 [x, v]
    
    返回:
        tuple: (振幅, 周期)
    """
    # 跳过初始瞬态，只分析稳态部分
    skip = int(len(states)*0.5)
    x = states[skip:, 0]  # 只分析位置x
    t = np.arange(len(x))  # 时间索引
    
    # 计算振幅（取最大值的平均）
    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:  # 寻找局部极大值
            peaks.append(x[i])
    amplitude = np.mean(peaks) if peaks else np.nan  # 计算平均振幅
    
    # 计算周期（取相邻峰值点的时间间隔平均）
    if len(peaks) >= 2:
        # 获取所有峰值点的时间索引
        peak_indices = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
        periods = np.diff(peak_indices)  # 计算相邻峰值点的时间差
        period = np.mean(periods) if len(periods) > 0 else np.nan
    else:
        period = np.nan
    
    return amplitude, period

def main():
    """主函数：执行模拟和分析"""
    # 基本参数设置
    mu = 1.0  # 非线性阻尼系数
    omega = 1.0  # 自然频率
    t_span = (0, 50)  # 模拟时间范围
    dt = 0.01  # 时间步长
    initial_state = np.array([1.0, 0.0])  # 初始状态 [x0, v0]
    
    # 任务1 - 基本实现
    t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t, states, f'Time Evolution of van der Pol Oscillator (μ={mu})')
    
    # 任务2 - 参数影响分析
    mu_values = [1.0, 2.0, 4.0]  # 不同的mu值
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_time_evolution(t, states, f'Time Evolution of van der Pol Oscillator (μ={mu})')
        amplitude, period = analyze_limit_cycle(states)
        print(f'μ = {mu}: Amplitude ≈ {amplitude:.3f}, Period ≈ {period*dt:.3f}')
    
    # 任务3 - 相空间分析
    for mu in mu_values:
        t, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_phase_space(states, f'Phase Space Trajectory of van der Pol Oscillator (μ={mu})')

if __name__ == "__main__":
    main()
