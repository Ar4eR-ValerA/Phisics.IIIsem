import json
from math import sin, cos, atan, pi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from abc import abstractmethod


class AngleCoordinate:
    def __init__(self, theta: float, phi: float, length: float, is_polarized: bool = True):
        self.is_polarized = is_polarized
        self.theta = theta
        self.phi = phi
        self.length = length


class DecartCoordinate:
    def __init__(self, x: float, y: float, z: float):
        self.z = z
        self.y = y
        self.x = x


def to_decart(angle: AngleCoordinate) -> DecartCoordinate:
    x = angle.length * sin(angle.theta) * cos(angle.phi)
    y = angle.length * sin(angle.theta) * sin(angle.phi)
    z = angle.length * cos(angle.theta)

    return DecartCoordinate(x, y, z)


def to_angle(decart: DecartCoordinate) -> AngleCoordinate:
    length = (decart.x ** 2 + decart.y ** 2 + decart.z ** 2) ** 0.5
    phi = 0
    if decart.y >= 0 and decart.x >= 0:
        phi = pi / 2 if decart.x == 0 else atan(decart.y / decart.x)
    elif decart.y >= 0 and decart.x < 0:
        phi = atan(decart.y / decart.x) + pi
    elif decart.y < 0 and decart.x >= 0:
        phi = -pi / 2 if decart.x == 0 else atan(decart.y / decart.x)
    elif decart.y < 0 and decart.x < 0:
        phi = atan(decart.y / decart.x) + pi

    theta = pi/2 if decart.z == 0 else abs(atan((decart.x ** 2 + decart.y ** 2) ** 0.5 / decart.z))

    return AngleCoordinate(theta, phi, length)


class Light:
    @abstractmethod
    def angle_delta(self, angle: AngleCoordinate, time: float, distance, phase) -> AngleCoordinate:
        pass

    @abstractmethod
    def get_angle(self, time: float, distance: float, phase: float) -> AngleCoordinate:
        pass

    @abstractmethod
    def get_wave_length(self) -> float:
        pass


class BasicLight(Light):
    def __init__(self, amplitude: float, wave_length: float):
        self.amplitude = amplitude
        self.wave_length = wave_length
        self.k = 2 * pi / wave_length
        self.frequency = 2 * pi * 299_792_458 / wave_length

    def angle_delta(self, angle: AngleCoordinate, time: float, distance, phase) -> AngleCoordinate:
        length = self.amplitude * cos(self.frequency * time - self.k * distance + phase) - angle.length
        return AngleCoordinate(-angle.theta, -angle.phi, length, False)

    def get_angle(self, time: float, distance, phase) -> AngleCoordinate:
        return AngleCoordinate(0, 0, 0, False)

    def get_wave_length(self) -> float:
        return self.wave_length


class FlatPolarizedLight(Light):
    def __init__(self, angle: float, light: Light):
        self.angle = angle
        self.light = light

    def angle_delta(self, angle: AngleCoordinate, time: float, distance, phase) -> AngleCoordinate:
        value = self.light.angle_delta(AngleCoordinate(pi / 2, 0, 0), time, distance, phase)
        return AngleCoordinate(value.theta, self.angle - angle.phi,
                               value.length * cos(self.angle - angle.phi) - angle.length)

    def get_angle(self, time: float, distance, phase) -> AngleCoordinate:
        angle_delta = self.light.angle_delta(AngleCoordinate(pi / 2, self.angle, 0), time, distance, phase)
        return AngleCoordinate(pi / 2, self.angle, angle_delta.length * cos(angle_delta.phi))

    def get_wave_length(self) -> float:
        return self.light.get_wave_length()


class TimeShiftedLight(Light):
    def __init__(self, light: Light, shift: float):
        self.light = light
        self.shift = shift

    def angle_delta(self, angle: AngleCoordinate, time: float, distance: float, phase: float) -> AngleCoordinate:
        return self.light.angle_delta(angle, time + self.shift, distance, phase)

    def get_angle(self, time: float, distance: float, phase: float) -> AngleCoordinate:
        return self.light.get_angle(time + self.shift, distance, phase)

    def get_wave_length(self) -> float:
        return self.light.get_wave_length()


class PhaseShiftedLight(Light):
    def __init__(self, light: Light, shift: float):
        self.light = light
        self.shift = shift

    def angle_delta(self, angle: AngleCoordinate, time: float, distance: float, phase: float) -> AngleCoordinate:
        return self.light.angle_delta(angle, time, distance, phase + self.shift)

    def get_angle(self, time: float, distance: float, phase: float) -> AngleCoordinate:
        return self.light.get_angle(time, distance, phase + self.shift)

    def get_wave_length(self) -> float:
        return self.light.get_wave_length()


class CompositeLight(Light):
    def __init__(self, first: Light, second: Light):
        self.first = first
        self.second = second

    def angle_delta(self, angle: AngleCoordinate, time: float, distance: float, phase: float) -> AngleCoordinate:
        my_angle = self.get_angle(time, distance, phase)
        return AngleCoordinate(my_angle.theta - angle.theta, my_angle.phi - angle.phi, my_angle.length - angle.length)

    def get_angle(self, time: float, distance: float, phase: float) -> AngleCoordinate:
        first_angle = self.first.get_angle(time, distance, phase)
        second_angle = self.second.get_angle(time, distance, phase)

        first_decart = to_decart(first_angle)
        second_decart = to_decart(second_angle)
        sum_decart = DecartCoordinate(first_decart.x + second_decart.x,
                                      first_decart.y + second_decart.y,
                                      first_decart.z + second_decart.z)

        return to_angle(sum_decart)

    def get_wave_length(self) -> float:
        if self.first.get_wave_length() != self.second.get_wave_length():
            raise Exception("Wave lengths must be equal")
        return self.first.get_wave_length()


class DistanceSplitedLight(Light):
    def __init__(self, distance: float, before: Light, after: Light):
        self.before = before
        self.after = after
        self.distance = distance

    def angle_delta(self, angle: AngleCoordinate, time: float, distance: float, phase: float) -> AngleCoordinate:
        return self.before.angle_delta(angle, time, distance, phase) \
            if distance < self.distance else self.after.angle_delta(angle, time, distance, phase)

    def get_angle(self, time: float, distance: float, phase: float) -> AngleCoordinate:
        return self.before.get_angle(time, distance, phase) \
            if distance < self.distance else self.after.get_angle(time, distance, phase)

    def get_wave_length(self) -> float:
        return self.before.get_wave_length()


class ExtraordinaryLight(Light):
    def __init__(self, angle: float, light: Light):
        self.light = light
        self.angle = angle

    def angle_delta(self, angle: AngleCoordinate, time: float, distance, phase) -> AngleCoordinate:
        delta = self.light.angle_delta(AngleCoordinate(pi / 2, self.angle + pi / 2, 0), time, distance, phase)
        return AngleCoordinate(pi / 2, self.angle - angle.phi, delta.length * sin(delta.phi) - angle.length)

    def get_angle(self, time: float, distance: float, phase: float) -> AngleCoordinate:
        delta = self.light.angle_delta(AngleCoordinate(pi / 2, self.angle + pi / 2, 0), time, distance, phase)
        return AngleCoordinate(pi / 2, self.angle, delta.length * sin(delta.phi))

    def get_wave_length(self) -> float:
        return self.light.get_wave_length()


class Polarizer:
    @abstractmethod
    def pass_light(self, light: Light) -> Light:
        pass

    @abstractmethod
    def get_distance(self) -> float:
        pass


class FlatPolarizer(Polarizer):
    def __init__(self, angle: float, distance: float = 0):
        self.distance = distance
        self.angle = angle

    def pass_light(self, light: Light) -> Light:
        return DistanceSplitedLight(self.distance, light, FlatPolarizedLight(self.angle, light))

    def get_distance(self) -> float:
        return self.distance


class PolarizationPlate(Polarizer):
    def __init__(self, angle: float, refraction_difference: float, width: float, distance: float = 0):
        self.angle = angle
        self.refraction_difference = refraction_difference
        self.width = width
        self.distance = distance

    def pass_light(self, light: Light) -> Light:
        phase_diff = 2 * pi * self.refraction_difference * self.width / light.get_wave_length()
        extraordinary_light = ExtraordinaryLight(self.angle - pi / 2, light)
        extraordinary_light = PhaseShiftedLight(extraordinary_light, phase_diff)
        ordinary_light = FlatPolarizedLight(self.angle, light)

        return DistanceSplitedLight(self.distance, light, CompositeLight(ordinary_light, extraordinary_light))

    def get_distance(self) -> float:
        return self.distance


class PolarizerComposition(Polarizer):
    def __init__(self, polarizers: [Polarizer]):
        self.polarizers = polarizers

    def pass_light(self, light: Light) -> Light:
        for polarizer in self.polarizers:
            light = polarizer.pass_light(light)

        return light

    def get_distance(self) -> float:
        return max(self.polarizers, key=lambda p: p.get_distance()).get_distance()


deserialized_polarizers: [Polarizer] = []

file = open("input.json", 'r')
data = json.load(file)
light_amplitude = float(data['light_amplitude'])
for flat_polarizer in data['flat_polarizers']:
    deserialized_polarizers.append(
        FlatPolarizer(float(flat_polarizer['angle']) * pi, float(flat_polarizer['distance'])))
for polarization_plate in data['polarization_plates']:
    deserialized_polarizers.append(PolarizationPlate(
        float(polarization_plate['angle'] * pi),
        float(polarization_plate['refraction_difference']),
        float(polarization_plate['width']),
        float(polarization_plate['distance'])))

deserialized_polarizers.sort(key=lambda p: p.get_distance())

max_polarizer_distance = deserialized_polarizers[-1].distance

passed_light: Light = PolarizerComposition(deserialized_polarizers).pass_light(BasicLight(light_amplitude, 500e-9))


def update(t):
    length = max_polarizer_distance * 1.2
    count = 2000
    ax.clear()
    ax.set_xlim3d(0, length)
    ax.set_ylim3d(-length, length)
    ax.set_zlim3d(-length, length)
    space = np.linspace(0, length, count)
    angles = [(x, passed_light.get_angle(t / (0.5 * 1e13), x, 0)) for x in space]
    angles = list(filter(lambda a: abs(a[1].length) > 1e-10 or not a[1].is_polarized, angles))
    decarts = [(a[0], to_decart(a[1])) for a in angles]
    decarts = [DecartCoordinate(d[1].x, d[1].y, d[1].z + d[0]) for d in decarts]

    for polarizer in deserialized_polarizers:
        d = polarizer.get_distance()
        x = [d] * 5
        y = [-max_polarizer_distance, -max_polarizer_distance, max_polarizer_distance, max_polarizer_distance,
             -max_polarizer_distance]
        z = [-max_polarizer_distance, max_polarizer_distance, max_polarizer_distance, -max_polarizer_distance,
             -max_polarizer_distance]

        ax.plot(x, y, z)

    x = []
    y = []
    z = []

    for d in decarts:
        x.append(d.z)
        y.append(-d.y)
        z.append(d.x)

    ax.plot(x, y, z)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ani = FuncAnimation(fig, update, 100000000, interval=100)
plt.show()
