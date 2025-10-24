"""
Pequeño programa de consola para trabajar con distribuciones:
- Binomial, Poisson (discretas)
- Normal, Exponencial (continuas)

No requiere librerías externas (solo 'math').
Incluye funciones PMF/PDF y CDF, y algunos cálculos típicos.
"""

from math import comb, factorial, exp, sqrt, erf

# =====================
# Distribuciones discretas
# =====================

def binomial_pmf(n: int, p: float, k: int) -> float:
    """P(X=k) para X ~ Binomial(n,p)."""
    if k < 0 or k > n:
        return 0.0
    q = 1.0 - p
    return comb(n, k) * (p ** k) * (q ** (n - k))

def binomial_cdf(n: int, p: float, k: int) -> float:
    """P(X<=k) sumando la PMF (definición)."""
    k = int(k)
    return sum(binomial_pmf(n, p, i) for i in range(0, k + 1))

def poisson_pmf(lmbda: float, k: int) -> float:
    """P(X=k) para X ~ Poisson(λ)."""
    if k < 0:
        return 0.0
    return (lmbda ** k) * exp(-lmbda) / factorial(k)

def poisson_cdf(lmbda: float, k: int) -> float:
    """P(X<=k) sumando la PMF (definición)."""
    k = int(k)
    return sum(poisson_pmf(lmbda, i) for i in range(0, k + 1))

# =====================
# Distribuciones continuas
# =====================

def normal_pdf(mu: float, sigma: float, x: float) -> float:
    """f(x) para X ~ N(mu, sigma^2)."""
    if sigma <= 0:
        raise ValueError("sigma debe ser > 0")
    z = (x - mu) / sigma
    return (1.0 / (sigma * sqrt(2.0 * 3.141592653589793))) * exp(-0.5 * z * z)

def normal_cdf(mu: float, sigma: float, x: float) -> float:
    """F(x) para X ~ N(mu, sigma^2) usando erf."""
    if sigma <= 0:
        raise ValueError("sigma debe ser > 0")
    z = (x - mu) / (sigma * sqrt(2.0))
    return 0.5 * (1.0 + erf(z))

def exponential_pdf(lmbda: float, t: float) -> float:
    """f(t) para T ~ Exp(λ)."""
    if lmbda <= 0:
        raise ValueError("lambda debe ser > 0")
    return lmbda * exp(-lmbda * t) if t >= 0 else 0.0

def exponential_cdf(lmbda: float, t: float) -> float:
    """F(t) para T ~ Exp(λ)."""
    if lmbda <= 0:
        raise ValueError("lambda debe ser > 0")
    return 1.0 - exp(-lmbda * t) if t >= 0 else 0.0

# =====================
# Utilidades de entrada/salida
# =====================

def ask_int(prompt: str, default=None):
    while True:
        s = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not s and default is not None:
            return int(default)
        try:
            return int(s)
        except ValueError:
            print("  > Ingresa un entero válido.")

def ask_float(prompt: str, default=None):
    while True:
        s = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not s and default is not None:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("  > Ingresa un número válido.")

def pause():
    input("\nPresiona Enter para continuar...")

# =====================
# Menús por distribución
# =====================

def menu_binomial():
    print("\n=== Binomial (n, p) ===")
    n = ask_int("n (ensayos)", 10)
    p = ask_float("p (prob. de éxito en cada ensayo)", 0.3)
    print("\nOpciones:\n  1) PMF: P(X=k)\n  2) CDF: P(X<=k)\n  3) Media y Varianza\n  4) Volver")
    while True:
        op = ask_int("Elige", 1)
        if op == 1:
            k = ask_int("k")
            print(f"P(X={k}) = {binomial_pmf(n,p,k):.6f}")
        elif op == 2:
            k = ask_int("k")
            print(f"P(X<={k}) = {binomial_cdf(n,p,k):.6f}")
        elif op == 3:
            mu = n * p
            var = n * p * (1.0 - p)
            print(f"E[X] = {mu:.6f}  |  Var(X) = {var:.6f}")
        else:
            break
        pause()

def menu_poisson():
    print("\n=== Poisson (lambda) ===")
    lmbda = ask_float("lambda (tasa media)", 2.5)
    print("\nOpciones:\n  1) PMF: P(X=k)\n  2) CDF: P(X<=k)\n  3) Media y Varianza\n  4) Volver")
    while True:
        op = ask_int("Elige", 1)
        if op == 1:
            k = ask_int("k")
            print(f"P(X={k}) = {poisson_pmf(lmbda,k):.6f}")
        elif op == 2:
            k = ask_int("k")
            print(f"P(X<={k}) = {poisson_cdf(lmbda,k):.6f}")
        elif op == 3:
            mu = lmbda
            var = lmbda
            print(f"E[X] = {mu:.6f}  |  Var(X) = {var:.6f}")
        else:
            break
        pause()

def menu_normal():
    print("\n=== Normal (mu, sigma) ===")
    mu = ask_float("mu", 0.0)
    sigma = ask_float("sigma (>0)", 1.0)
    print("\nOpciones:\n  1) PDF: f(x)\n  2) CDF: F(x)\n  3) Media y Varianza\n  4) Volver")
    while True:
        op = ask_int("Elige", 1)
        if op == 1:
            x = ask_float("x")
            print(f"f({x}) = {normal_pdf(mu,sigma,x):.6f}")
        elif op == 2:
            x = ask_float("x")
            print(f"F({x}) = {normal_cdf(mu,sigma,x):.6f}")
        elif op == 3:
            print(f"E[X] = {mu:.6f}  |  Var(X) = {sigma*sigma:.6f}")
        else:
            break
        pause()

def menu_exponencial():
    print("\n=== Exponencial (lambda) ===")
    lmbda = ask_float("lambda (>0)", 1.0)
    print("\nOpciones:\n  1) PDF: f(t)\n  2) CDF: F(t)\n  3) Media y Varianza\n  4) Volver")
    while True:
        op = ask_int("Elige", 1)
        if op == 1:
            t = ask_float("t (>=0)", 0.0)
            print(f"f({t}) = {exponential_pdf(lmbda,t):.6f}")
        elif op == 2:
            t = ask_float("t (>=0)", 0.0)
            print(f"F({t}) = {exponential_cdf(lmbda,t):.6f}")
        elif op == 3:
            mu = 1.0 / lmbda
            var = 1.0 / (lmbda * lmbda)
            print(f"E[T] = {mu:.6f}  |  Var(T) = {var:.6f}")
        else:
            break
        pause()

# =====================
# Menú principal
# =====================

def main():
    while True:
        print("\n==============================")
        print("   Probabilidad – Consola")
        print("==============================")
        print("1) Binomial")
        print("2) Poisson")
        print("3) Normal")
        print("4) Exponencial")
        print("5) Salir")
        op = ask_int("Elige", 1)
        if op == 1:
            menu_binomial()
        elif op == 2:
            menu_poisson()
        elif op == 3:
            menu_normal()
        elif op == 4:
            menu_exponencial()
        else:
            print("¡Hasta luego!")
            break

if __name__ == "__main__":
    main()
