import math
import matplotlib.pyplot as plt
import pygame
from numpy import linspace
from scipy.integrate import odeint


class SystemParameters:
    def __init__(self,
                 moment: float,
                 distance: float,
                 mass: float,
                 start_velocity: float,
                 start_phase: float,
                 force: float,
                 force_frequency: float,
                 force_start_phase: float):
        self.moment = moment
        self.distance = distance
        self.mass = mass
        self.start_velocity = start_velocity
        self.start_phase = start_phase
        self.force = force
        self.force_frequency = force_frequency
        self.force_start_phase = force_start_phase


def func(u, time, params: SystemParameters):
    y, z = u
    dz = params.force / params.mass * math.cos(
        params.force_frequency * time + params.force_start_phase) - params.mass * 9.8 * params.distance / params.moment * y

    return [z, dz]


solutions: [float] = []
current = 0


def get_angle(params: SystemParameters, time: float, time_span: float) -> float:
    if len(solutions) <= current:
        space = linspace(time, time + 120, int(120 / time_span))
        u = odeint(func, [params.start_phase, params.start_velocity], space, args=tuple([params]))
        y = u[:, 0]
        for value in y:
            solutions.append(value)

    return solutions[current]


parameters = SystemParameters(
    float(input("Moment: ")),
    float(input("Distance: ")),
    float(input("Mass: ")),
    float(input("Start velocity: ")),
    float(input("Start phase: ")),
    float(input("Force: ")),
    float(input("Force frequency: ")),
    float(input("Force start phase: ")))

pygame.init()
width = 1500
height = 900
win = pygame.display.set_mode([width, height])
pygame.display.set_caption('Моделирование №1: Физический маятник (свободные и вынужденные колебания)')
clock = pygame.time.Clock()
fps = 60
pygame.display.flip()
font_size = 32
font = pygame.font.SysFont('serif', font_size)
t = 0
t_s = 0.02

times = []
angles = []

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            running = False

    if not running:
        break

    clock.tick(fps)
    win.fill((255, 255, 255))

    s_x = width / 2
    s_y = height * 0.2
    length = height / 2.5
    angle = get_angle(parameters, t, t_s)
    current += 1
    times.append(t)
    angles.append(angle)
    m_x = s_x + length * math.sin(angle)
    m_y = s_y + length * math.cos(angle)

    pygame.draw.circle(win, (0, 0, 0), (s_x, s_y), height / 40)
    pygame.draw.line(win, (0, 0, 0), (s_x, s_y), (m_x, m_y))
    pygame.draw.circle(win, (255, 0, 0), (m_x, m_y), height / 15)

    pygame.display.update()
    t += t_s

plt.plot(times, angles)
plt.title("⍺(t)")
plt.xlabel("t, c")
plt.ylabel("⍺, radian")
plt.show()