import asyncio
from bleak import BleakScanner
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import deque
import random

def rssi_to_distance(rssi, tx_power=-59, n=2):
    return 10 ** ((tx_power - rssi) / (10 * n))

def triangulate(beacon_data, beacon_positions):
    if len(beacon_data) < 3:
        return None

    distances = [rssi_to_distance(beacon['rssi']) for beacon in beacon_data]
    weights = [1/d for d in distances]
    x, y = np.average([pos[0] for pos in beacon_positions], weights=weights), np.average([pos[1] for pos in beacon_positions], weights=weights)
    return x, y

async def scan_and_collect_positions():
    scanner = BleakScanner()
    beacon_positions = {
        'C3:00:00:1C:6E:69': (0, 0),
        'C3:00:00:1C:6E:60': (3.5, 0),
        'C3:00:00:1C:6E:6F': (7, 0),
        'C3:00:00:1C:6E:6A': (0, 5),
        'C3:00:00:1C:6E:5E': (3.5, 5),
        'C3:00:00:1C:6E:5F': (7, 5)
    }
    collected_positions = []

    while len(collected_positions) < 10:
        devices = await scanner.discover()
        beacon_data = []

        for device in devices:
            if device.name == "MBeacon" and device.address in beacon_positions:
                beacon_data.append({'rssi': device.rssi, 'address': device.address})

                if len(beacon_data) >= 3:
                    positions = [beacon_positions[data['address']] for data in beacon_data]
                    estimated_position = triangulate(beacon_data, positions)
                    if estimated_position:
                        collected_positions.append(estimated_position)
                        print(f"Collected position: {estimated_position}")

        await asyncio.sleep(1)

    avg_x = np.mean([pos[0] for pos in collected_positions])
    avg_y = np.mean([pos[1] for pos in collected_positions])
    print(f"Average position: ({avg_x}, {avg_y})")
    return (avg_x, avg_y)

# 주차 공간의 범위 정의
parking_spots = [
    (0, 2, 1, 2), (2.5, 4.5, 1, 2), (5, 7, 1, 2),
    (0, 2, 2, 3), (2.5, 4.5, 2, 3), (5, 7, 2, 3),
    (0, 2, 3, 4), (2.5, 4.5, 3, 4), (5, 7, 3, 4)
]

# 차도 범위 정의
path_points = [
    (0, 0.5, 0.5, 1), (0.5, 1, 0.5, 1), (1, 1.5, 0.5, 1), (1.5, 2, 0.5, 1), (2, 2.5, 0.5, 1),
    (2.5, 3, 0.5, 1), (3, 3.5, 0.5, 1), (3.5, 4, 0.5, 1), (4, 4.5, 0.5, 1), (2, 2.5, 2.5, 3),
    (4.5, 5, 0.5, 1), (5, 5.5, 0.5, 1), (5.5, 6, 0.5, 1), (6, 6.5, 0.5, 1), (6.5, 7, 0.5, 1),
    (2, 2.5, 3.5, 4), (2, 2.5, 2, 2.5), (2, 2.5, 3, 3.5), (2, 2.5, 1, 1.5), (2, 2.5, 1.5, 2),
    (4.5, 5, 3.5, 4), (4.5, 5, 2, 2.5), (4.5, 5, 3, 3.5), (4.5, 5, 1, 1.5), (4.5, 5, 1.5, 2),
    (0, 0.5, 4, 4.5), (0.5, 1, 4, 4.5), (1, 1.5, 4, 4.5), (1.5, 2, 4, 4.5), (2, 2.5, 4, 4.5),
    (2.5, 3, 4, 4.5), (3, 3.5, 4, 4.5), (3.5, 4, 4, 4.5), (4, 4.5, 4, 4.5), (4.5, 5, 2.5, 3),
    (4.5, 5, 4, 4.5), (5, 5.5, 4, 4.5), (5.5, 6, 4, 4.5), (6, 6.5, 4, 4.5), (6.5, 7, 4, 4.5)
]

# 중앙 좌표 계산 함수
def mid_coord(x_min, x_max, y_min, y_max):
    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
    return (x, y)

# 주차 공간 중앙 좌표 딕셔너리
parking_spots_mid = {spot: mid_coord(*spot) for spot in parking_spots}

# 차도 중앙 좌표 딕셔너리
path_points_mid = {path_rect: mid_coord(*path_rect) for path_rect in path_points}

async def main():
    current_position = await scan_and_collect_positions()

    # 빈 주차 공간 설정
    empty_spots = list(parking_spots_mid.values())

    # 빈 주차 공간 중 랜덤하게 두 개 선택
    random_empty_spots = random.sample(empty_spots, 2)

    # 가장 가까운 빈 주차 공간 찾기
    closest_spot = min(random_empty_spots, key=lambda spot: distance.euclidean(current_position, spot))


    def map_spot(spot):
        mappings = {
            (1.0, 1.5): (2.25, 1.5), (3.5, 1.5): (4.75, 1.5), (6.0, 1.5): (4.75, 1.5),
            (1.0, 2.5): (2.25, 2.5), (3.5, 2.5): (4.75, 2.5), (6.0, 2.5): (4.75, 2.5),
            (1.0, 3.5): (2.25, 3.5), (3.5, 3.5): (4.75, 3.5), (6.0, 3.5): (4.75, 3.5)
        }
        return mappings.get(spot, spot)

    mapped_closest_spot = map_spot(closest_spot)


    path_points_transformed = list(path_points_mid.values())


    mapped_green_spot = min(path_points_transformed, key=lambda spot: distance.euclidean(mapped_closest_spot, spot))

    # BFS 알고리즘을 사용하여 경로 찾기
    def bfs(start, goal, graph):
        queue = deque([start])
        came_from = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            for neighbor in get_neighbors(current, graph):
                if neighbor not in came_from:
                    queue.append(neighbor)
                    came_from[neighbor] = current
        return []

    def get_neighbors(node, graph):
        neighbors = []
        for point in graph:
            if point != node:
                if (abs(point[0] - node[0]) <= 0.5 and point[1] == node[1]) or (abs(point[1] - node[1]) <= 0.5 and point[0] == node[0]):
                    neighbors.append(point)
        return neighbors

    def define_edges(graph):
        edges = {}
        for point in graph:
            edges[point] = get_neighbors(point, graph)
        return edges

    edges = define_edges(path_points_transformed)

    # 경로 찾기
    path = bfs(current_position, mapped_green_spot, path_points_transformed)

    # 그래프로 시각화
    plt.figure(figsize=(10, 7))

    # 주차 공간 그리기
    for spot in parking_spots:
        plt.gca().add_patch(plt.Rectangle((spot[0], spot[2]), spot[1] - spot[0], spot[3] - spot[2], edgecolor='black', facecolor='none'))

    # 차도 그리기
    for path_rect in path_points:
        plt.gca().add_patch(plt.Rectangle((path_rect[0], path_rect[2]), path_rect[1] - path_rect[0], path_rect[3] - path_rect[2], edgecolor='none', facecolor='gray'))

    # 노드와 엣지 그리기
    for node, neighbors in edges.items():
        plt.scatter(*node, color='black', s=50) 
        for neighbor in neighbors:
            plt.plot([node[0], neighbor[0]], [node[1], neighbor[1]], color='blue') 

    # 현재 위치와 빈 주차 공간 그리기
    for spot in random_empty_spots:
        plt.scatter(*spot, color='red', s=100)  
    plt.scatter(*current_position, color='blue', s=100)  
    plt.scatter(*mapped_green_spot, color='green', s=100) 

    # 경로 그리기
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, color='red') 
        for i in range(len(path) - 1):
            plt.arrow(path_x[i], path_y[i], path_x[i+1] - path_x[i], path_y[i+1] - path_y[i], head_width=0.1, length_includes_head=True, color='blue')  # 화살표로 경로 표시
    else:
        print("No path found.")

    # 좌표 설정 및 보여주기
    plt.xlim(0, 7)
    plt.ylim(0, 5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Parking Lot Path Finding')
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())
