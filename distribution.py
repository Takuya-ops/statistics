import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D


# ポアソン分布
def plot_poisson_distribution(lam):
    x = np.arange(0, 20)
    y = stats.poisson.pmf(x, lam)

    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.title(f"ポアソン分布 (λ={lam})")
    plt.xlabel("事象の発生回数")
    plt.ylabel("確率")
    plt.show()


# ガンマ分布
def plot_gamma_distribution(k, theta):
    x = np.linspace(0, 20, 200)
    y = stats.gamma.pdf(x, a=k, scale=theta)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(f"ガンマ分布 (k={k}, θ={theta})")
    plt.xlabel("値")
    plt.ylabel("確率密度")
    plt.show()


# ベータ分布
def plot_beta_distribution(a, b):
    x = np.linspace(0, 1, 200)
    y = stats.beta.pdf(x, a, b)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(f"ベータ分布 (α={a}, β={b})")
    plt.xlabel("値")
    plt.ylabel("確率密度")
    plt.show()


# コーシー分布
def plot_cauchy_distribution(x0, gamma):
    x = np.linspace(-10, 10, 200)
    y = stats.cauchy.pdf(x, loc=x0, scale=gamma)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(f"コーシー分布 (x0={x0}, γ={gamma})")
    plt.xlabel("値")
    plt.ylabel("確率密度")
    plt.show()


# 対数正規分布
def plot_lognormal_distribution(mu, sigma):
    x = np.linspace(0, 10, 200)
    y = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(f"対数正規分布 (μ={mu}, σ={sigma})")
    plt.xlabel("値")
    plt.ylabel("確率密度")
    plt.show()


# 2変量正規分布
def plot_bivariate_normal_distribution(mu1, mu2, sigma1, sigma2, rho):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = stats.multivariate_normal(
        [mu1, mu2],
        [[sigma1**2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2**2]],
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, rv.pdf(pos), cmap="viridis")
    ax.set_title(
        f"2変量正規分布 (μ1={mu1}, μ2={mu2}, σ1={sigma1}, σ2={sigma2}, ρ={rho})"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("確率密度")
    plt.show()


# 混合正規分布
def plot_mixture_normal_distribution(mu1, sigma1, mu2, sigma2, weight):
    x = np.linspace(-10, 10, 200)
    y1 = stats.norm.pdf(x, mu1, sigma1)
    y2 = stats.norm.pdf(x, mu2, sigma2)
    y_mix = weight * y1 + (1 - weight) * y2

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_mix, label="混合分布")
    plt.plot(x, y1, "--", label="成分1")
    plt.plot(x, y2, "--", label="成分2")
    plt.title(
        f"混合正規分布 (μ1={mu1}, σ1={sigma1}, μ2={mu2}, σ2={sigma2}, w={weight})"
    )
    plt.xlabel("値")
    plt.ylabel("確率密度")
    plt.legend()
    plt.show()


# カイ二乗分布
def plot_chi_square_distribution(df):
    x = np.linspace(0, 20, 200)
    y = stats.chi2.pdf(x, df)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(f"カイ二乗分布 (自由度={df})")
    plt.xlabel("値")
    plt.ylabel("確率密度")
    plt.show()


# t分布
def plot_t_distribution(df):
    x = np.linspace(-5, 5, 200)
    y = stats.t.pdf(x, df)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(f"t分布 (自由度={df})")
    plt.xlabel("値")
    plt.ylabel("確率密度")
    plt.show()


# F分布
def plot_f_distribution(dfn, dfd):
    x = np.linspace(0, 5, 200)
    y = stats.f.pdf(x, dfn, dfd)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(f"F分布 (自由度1={dfn}, 自由度2={dfd})")
    plt.xlabel("値")
    plt.ylabel("確率密度")
    plt.show()


def plot_geometric_distribution(p, n):
    x = np.arange(1, n + 1)
    y = stats.geom.pmf(x, p)

    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.title(f"幾何分布 (p={p})")
    plt.xlabel("試行回数")
    plt.ylabel("確率")
    plt.show()


def plot_binomial_distribution(n, p):
    x = np.arange(0, n + 1)
    y = stats.binom.pmf(x, n, p)

    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.title(f"二項分布 (n={n}, p={p})")
    plt.xlabel("成功回数")
    plt.ylabel("確率")
    plt.show()


def plot_poisson_distribution(lam):
    x = np.arange(0, 20)
    y = stats.poisson.pmf(x, lam)

    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.title(f"ポアソン分布 (λ={lam})")
    plt.xlabel("事象の発生回数")
    plt.ylabel("確率")
    plt.show()


def plot_normal_distribution(mu, sigma):
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    y = stats.norm.pdf(x, mu, sigma)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(f"正規分布 (μ={mu}, σ={sigma})")
    plt.xlabel("値")
    plt.ylabel("確率密度")
    plt.show()


def plot_exponential_distribution(lam):
    x = np.linspace(0, 5 / lam, 100)
    y = stats.expon.pdf(x, scale=1 / lam)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(f"指数分布 (λ={lam})")
    plt.xlabel("時間")
    plt.ylabel("確率密度")
    plt.show()


# 例：6の目が出るまでサイコロを振る回数の分布
plot_geometric_distribution(1 / 6, 20)
# 例：10回コインを投げて表が出る回数の分布
plot_binomial_distribution(10, 0.5)
# 例：1時間あたりの平均顧客数が5人の場合の分布
plot_poisson_distribution(5)
# 例：平均身長170cm、標準偏差5cmの身長分布
plot_normal_distribution(170, 5)
# 例：平均故障間隔が2年の機器の故障間隔分布
plot_exponential_distribution(1 / 2)

plot_poisson_distribution(3)
plot_gamma_distribution(2, 2)
plot_beta_distribution(2, 5)
plot_cauchy_distribution(0, 1)
plot_lognormal_distribution(0, 0.5)
plot_bivariate_normal_distribution(0, 0, 1, 1, 0.5)
plot_mixture_normal_distribution(-2, 1, 2, 1, 0.3)
plot_chi_square_distribution(3)
plot_t_distribution(5)
plot_f_distribution(5, 10)
