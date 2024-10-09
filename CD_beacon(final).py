import asyncio
from bleak import BleakScanner
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime

class KalmanFilter:
    def __init__(self, process_noise=0.005, measurement_noise=20):
        self.initialized = False
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.predicted_rssi = 0
        self.error_covariance = 1  # 초기 오차 공분산을 1로 설정

    def filter(self, rssi): 
        if not self.initialized:
            self.initialized = True
            prior_rssi = rssi
            prior_error_covariance = 1
        else:
            prior_rssi = self.predicted_rssi
            prior_error_covariance = self.error_covariance + self.process_noise

        kalman_gain = prior_error_covariance / (prior_error_covariance + self.measurement_noise)
        self.predicted_rssi = prior_rssi + kalman_gain * (rssi - prior_rssi)
        self.error_covariance = (1 - kalman_gain) * prior_error_covariance

        # 동적으로 측정 노이즈와 과정 노이즈를 업데이트
        self.process_noise = max(self.process_noise * 0.9, 0.001)  # 최소 값으로 제한
        self.measurement_noise = max(self.measurement_noise * 0.9, 1)  # 최소 값으로 제한

        return self.predicted_rssi

def rssi_to_distance(rssi, tx_power=-33, n=4):
    return 10 ** ((tx_power - rssi) / (10 * n))

def triangulate(beacon_data, beacon_positions):
    if len(beacon_data) < 3:
        return None

    distances = [rssi_to_distance(beacon['rssi']) for beacon in beacon_data]
    weights = [1 / d for d in distances]
    total_weight = sum(weights)

    x = sum(pos[0] * weight for pos, weight in zip(beacon_positions, weights)) / total_weight
    y = sum(pos[1] * weight for pos, weight in zip(beacon_positions, weights)) / total_weight
    
    return x, y

def display_points_on_graph(positions, filename=None):
    plt.figure(figsize=(10, 6))
    for pos in positions:
        plt.scatter(*pos, color='red')
        plt.text(pos[0], pos[1], f' ({pos[0]}, {pos[1]})', color='blue', fontsize=12)
    plt.xlim(0, 15)
    plt.ylim(0, 10)
    title = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    plt.title(f'Location Plot {title}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    
    if filename is None:
        filename = datetime.now().strftime("location_plot_%Y%m%d_%H%M%S.png")
    
    plt.savefig(filename)
    plt.close()

    image = Image.open(filename)
    image.show()

async def scan_and_display():
    scanner = BleakScanner()
    kalman_filters = {}

    beacon_positions = {
        'C3:00:00:1A:AC:34': (0, 0),
        'C3:00:00:1A:AC:35': (0, 3),
        'C3:00:00:1A:AC:61': (3, 0),
        'C3:00:00:1A:AC:60': (3, 3),
        'C3:00:00:1A:AC:5A': (6, 0),
        'C3:00:00:1A:AC:5B': (6, 3)
    }

    collected_positions = []

    while len(collected_positions) < 10:
        devices = await scanner.discover()
        print("인식된 비콘: ", devices)
        beacon_data = {}

        raw_rssi_values = []
        stable_rssi_values = []
        distance_values = []
        beacon_addresses = []

        for device in devices:
            if device.name == "MBeacon" and device.address in beacon_positions:
                if device.address not in kalman_filters:
                    kalman_filters[device.address] = KalmanFilter()

                filtered_rssi = kalman_filters[device.address].filter(device.rssi)
                beacon_data[device.address] = {'rssi': filtered_rssi, 'address': device.address}
                raw_rssi_values.append(device.rssi)
                stable_rssi_values.append(filtered_rssi)
                distance_values.append(rssi_to_distance(filtered_rssi))
                beacon_addresses.append(device.address)

                if len(beacon_data) >= 3:
                    unique_beacon_data = list(beacon_data.values())
                    positions = [beacon_positions[data['address']] for data in unique_beacon_data]
                    estimated_position = triangulate(unique_beacon_data, positions)

                    if estimated_position:
                        collected_positions.append(estimated_position)
                        print(f"Collected position: {estimated_position}")
                        print(f"Raw RSSI values: {raw_rssi_values}")
                        print(f"Filtered RSSI values: {stable_rssi_values}")
                        print(f"Distances: {distance_values}")
                        print(f"Beacons: {beacon_addresses}")

                        if len(collected_positions) >= 10:
                            break

                    break

    if collected_positions:
        display_points_on_graph(collected_positions)

def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(scan_and_display())

if __name__ == "__main__":
    main()
