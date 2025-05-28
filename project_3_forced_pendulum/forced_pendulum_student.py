import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def forced_pendulum_ode(t, state, l, g, C, Omega):
    """
    受驱单摆的常微分方程
    state: [theta, omega]
    返回: [dtheta/dt, domega/dt]
    """
    # TODO: 在此实现受迫单摆的ODE方程
    theta, omega = state
    dtheta = omega
    domega = -(g/l)*np.sin(theta) - (C)*omega + np.cos(Omega*t)
    return [dtheta, domega]

def solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, t_span=(0,100), y0=[0,0]):
    """
    求解受迫单摆运动方程
    返回: t, theta
    """
    # TODO: 使用solve_ivp求解受迫单摆方程
    # 提示: 需要调用forced_pendulum_ode函数
    sol = solve_ivp(forced_pendulum_ode, t_span, y0, 
                    args=(l, g, C, Omega), 
                    method='RK45', 
                    dense_output=True)
    t = np.linspace(t_span[0], t_span[1], 1000)
    theta = sol.sol(t)[0]
    return t, theta


def find_resonance(l=0.1, g=9.81, C=2, Omega_range=None, t_span=(0,200), y0=[0,0]):
    """
    寻找共振频率
    返回: Omega_range, amplitudes
    """
    # TODO: 实现共振频率查找功能
    # 提示: 需要调用solve_pendulum函数并分析结果
    if Omega_range is None:
        Omega_range = np.linspace(0.1, 5, 50)
    
    amplitudes = []
    for Omega in Omega_range:
        t, theta = solve_pendulum(l, g, C, Omega, t_span, y0)
        # 取最后1/4周期的振幅作为稳态振幅
        steady_state = theta[len(theta)//4*3:]
        amplitude = (np.max(steady_state) - np.min(steady_state))/2
        amplitudes.append(amplitude)
    
    return Omega_range, np.array(amplitudes)
def plot_results(t, theta, title):
    """绘制结果"""
    # 此函数已提供完整实现，学生不需要修改
    plt.figure(figsize=(10, 5))
    plt.plot(t, theta)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.show()

def main():
    """主函数"""
    # 任务1: 特定参数下的数值解与可视化
    # TODO: 调用solve_pendulum和plot_results
    t, theta = solve_pendulum()
    plot_results(t, theta, "Forced Pendulum Motion (Ω=5 rad/s)")
    # 任务2: 探究共振现象
    # TODO: 调用find_resonance并绘制共振曲线
    Omega_range, amplitudes = find_resonance()
    plt.figure(figsize=(10, 5))
    plt.plot(Omega_range, amplitudes)
    plt.title("Resonance Curve")
    plt.xlabel("Driving Frequency (rad/s)")
    plt.ylabel("Amplitude (rad)")
    plt.grid(True)
    plt.show()
    # 找到共振频率并绘制共振情况
    # TODO: 实现共振频率查找和绘图
    resonance_Omega = Omega_range[np.argmax(amplitudes)]
    print(f"Resonance frequency: {resonance_Omega:.2f} rad/s")
    t, theta = solve_pendulum(Omega=resonance_Omega)
    plot_results(t, theta, f"Pendulum at Resonance (Ω={resonance_Omega:.2f} rad/s)")
if __name__ == '__main__':
    main()
