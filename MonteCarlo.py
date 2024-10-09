import numpy as np
import matplotlib.pyplot as plt


def estimate_pi(n):
    points_inside_circle = 0
    total_points = n

    x_inside = []
    y_inside = []
    x_outside = []
    y_outside = []

    for _ in range(total_points):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            points_inside_circle += 1
            x_inside.append(x)
            y_inside.append(y)
        else:
            x_outside.append(x)
            y_outside.append(y)

    pi_estimate = 4 * points_inside_circle / total_points
    return pi_estimate, x_inside, y_inside, x_outside, y_outside


# パラメータ設定
n_points = 10000  # 点の数を減らして処理を軽くする

# πの推定とデータ点の取得
estimated_pi, x_inside, y_inside, x_outside, y_outside = estimate_pi(n_points)

# プロットの作成
plt.figure(figsize=(10, 10))
plt.scatter(x_inside, y_inside, c="blue", alpha=0.1, label="Inside")
plt.scatter(x_outside, y_outside, c="red", alpha=0.1, label="Outside")

# 円の描画
circle = plt.Circle((0, 0), 1, fill=False, color="green")
plt.gca().add_artist(circle)

plt.axis("equal")
plt.title(
    f"Monte Carlo Method to Estimate π\nEstimated π = {estimated_pi:.6f}, True π = {np.pi:.6f}"
)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# 結果の表示
print(f"Estimated π: {estimated_pi:.6f}")
print(f"True π: {np.pi:.6f}")
print(f"Error: {abs(estimated_pi - np.pi):.6f}")

plt.show()
