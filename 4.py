import numpy as np
import random

# Матрица расстояний
costMatrix = [
    [float('inf'), 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
    [2451, float('inf'), 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
    [713, 1745, float('inf'), 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
    [1018, 1524, 355, float('inf'), 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
    [1631, 831, 920, 700, float('inf'), 663, 1021, 1769, 949, 796, 879, 586, 371],
    [1374, 1240, 803, 862, 663, float('inf'), 1681, 1551, 1765, 547, 225, 887, 999],
    [2408, 959, 1737, 1395, 1021, 1681, float('inf'), 2493, 678, 1724, 1891, 1114, 701],
    [213, 2596, 851, 1123, 1769, 1551, 2493, float('inf'), 2699, 1038, 1605, 2300, 2099],
    [2571, 403, 1858, 1584, 949, 1765, 678, 2699, float('inf'), 1744, 1645, 653, 600],
    [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, float('inf'), 679, 1272, 1162],
    [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, float('inf'), 1017, 1200],
    [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, float('inf'), 504],
    [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, float('inf')],
]

N = len(costMatrix)  # Количество городов

# Муравьиный алгоритм
def ant_colony_optimization(costMatrix, num_ants, num_iterations, alpha, beta, rho, Q):
    num_cities = len(costMatrix) # количество городов
    pheromones = np.ones((num_cities, num_cities)) / num_cities # матрица единиц / количество городов
    best_distance = float('inf')
    best_path = []

    for iteration in range(num_iterations):
        paths = []
        path_lengths = []

        for ant in range(num_ants):
            path = [random.randint(0, num_cities - 1)]
            while len(path) < num_cities:
                i = path[-1] # текущая точка маршрута
                probabilities = [] # вероятности
                for j in range(num_cities):
                    if j not in path:
                        probabilities.append(
                            (pheromones[i][j] ** alpha) * ((1 / costMatrix[i][j]) ** beta)
                        )
                    else:
                        probabilities.append(0)
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()
                next_city = np.random.choice(range(num_cities), p=probabilities) #выбор следующего города в зависимости от вероятности перехода на этот город
                path.append(next_city)

            # Закрыть цикл
            path_lengths.append(
                sum(costMatrix[path[i - 1]][path[i]] for i in range(1, len(path)))
                + costMatrix[path[-1]][path[0]]
            )
            paths.append(path)

        # Обновление феромонов
        pheromones *= (1 - rho)
        for path, length in zip(paths, path_lengths):
            for i in range(len(path)):
                pheromones[path[i - 1], path[i]] += Q / length

        # Проверка лучшего решения
        min_length = min(path_lengths)
        if min_length < best_distance:
            best_distance = min_length
            best_path = paths[path_lengths.index(min_length)]

    return best_path, best_distance

# Настройки алгоритма
num_ants = 20
num_iterations = 100
alpha = 1  # Влияние феромона (Большие значения alpha делают алгоритм более зависимым от уже существующих феромонов )
beta = 2   # Влияние эвристики (в данном случае, обратного расстояния) на вероятность выбора ребра. Большие значения beta делают алгоритм более жадным, выбирающим более короткие ребра.
rho = 0.5  # Испарение феромона (Определяет, как быстро феромоны исчезают с ребер.)
Q = 100  # Параметр, определяющий количество феромонов, добавляемых на ребро после прохождения по нему муравья.

# Запуск
best_path, best_distance = ant_colony_optimization(costMatrix, num_ants, num_iterations, alpha, beta, rho, Q)
print("Лучший путь:", best_path)
print("Длина пути:", best_distance)
