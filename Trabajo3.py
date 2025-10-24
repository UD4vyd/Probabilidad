"""
CLI para resolver problemas de probabilidad conjunta (discreto y continuo).
- Interfaz en consola agradable con 'rich'.
- Cálculos discretos con numpy.
- Cálculos continuos (integrales simbólicas) con sympy.
Autor: ChatGPT
"""

from __future__ import annotations
import sys
from typing import List, Tuple, Optional

# Dependencias externas
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.text import Text
from rich import box

import numpy as np
import sympy as sp

console = Console()

# =============================================================
# Utilidades comunes
# =============================================================

def banner():
    title = Text("Probabilidad Conjunta CLI", style="bold cyan")
    subtitle = Text("Discreto (PMF) y Continuo (PDF)", style="magenta")
    console.print(Panel.fit(Text.assemble(title, "\n", subtitle), box=box.HEAVY))
    console.print("[dim]Autor: ChatGPT · Licencia: MIT[/dim]\n")

def pause():
    console.print("\n[dim]Presiona Enter para continuar...[/dim]")
    try:
        input()
    except EOFError:
        pass

def show_table(matrix: np.ndarray, row_labels: List[str], col_labels: List[str], title="Tabla"):
    table = Table(title=title, box=box.MINIMAL_DOUBLE_HEAD, header_style="bold cyan")
    table.add_column("x\\y", style="bold")
    for c in col_labels:
        table.add_column(c, justify="right")
    for i, r in enumerate(row_labels):
        row = [r] + [f"{matrix[i,j]:.6f}" for j in range(matrix.shape[1])]
        table.add_row(*row)
    console.print(table)

def pretty_expr(expr: sp.Expr, prefix: str=""):
    console.print(Panel.fit(Text(f"{prefix}{sp.sstr(expr)}", style="green"), title="Expresión simbólica", box=box.ROUNDED))

# =============================================================
# Caso DISCRETO
# =============================================================

def input_discrete() -> Tuple[np.ndarray, List[float], List[float]]:
    console.rule("[bold]Caso discreto (PMF)")
    m = IntPrompt.ask("Cantidad de valores de X (filas)", default=6)
    n = IntPrompt.ask("Cantidad de valores de Y (columnas)", default=6)
    console.print("\nValores de X y Y (por defecto 1..m y 1..n).", style="dim")
    default_x_vals = list(range(1, m+1))
    default_y_vals = list(range(1, n+1))
    x_vals_str = Prompt.ask(f"Valores de X separados por coma", default=",".join(map(str, default_x_vals)))
    y_vals_str = Prompt.ask(f"Valores de Y separados por coma", default=",".join(map(str, default_y_vals)))
    x_vals = [float(v.strip()) for v in x_vals_str.split(",")]
    y_vals = [float(v.strip()) for v in y_vals_str.split(",")]
    if len(x_vals) != m or len(y_vals) != n:
        console.print("[red]La cantidad de valores no coincide con m×n.[/red] Uso 1..m y 1..n por defecto.")
        x_vals = [float(i) for i in default_x_vals]
        y_vals = [float(i) for i in default_y_vals]

    console.print("\nElige cómo cargar la PMF conjunta p_{X,Y}(x_i,y_j):", style="bold")
    console.print("  [1] Uniforme (todos iguales y normalizados)")
    console.print("  [2] Cargar manualmente cada celda")
    console.print("  [3] Ejemplo de dados justos (6×6, 1/36)")

    choice = IntPrompt.ask("Opción", choices=["1","2","3"], default= "3")
    if choice == 1:
        P = np.ones((m,n), dtype=float)
        P = P / P.sum()
    elif choice == 2:
        P = np.zeros((m,n), dtype=float)
        console.print("Introduce las probabilidades celda por celda (sugerido que sumen 1):", style="dim")
        for i in range(m):
            for j in range(n):
                P[i,j] = FloatPrompt.ask(f"p(X={x_vals[i]}, Y={y_vals[j]})", default=0.0)
        s = P.sum()
        if s <= 0:
            console.print("[red]La suma es 0. Normalizo a uniforme.[/red]")
            P = np.ones((m,n), dtype=float) / (m*n)
        else:
            if abs(s-1.0) > 1e-10:
                console.print(f"[yellow]Advertencia:[/yellow] la suma es {s:.6f} ≠ 1. Normalizando...")
                P = P / s
    else:
        m, n = 6, 6
        x_vals = list(range(1,7))
        y_vals = list(range(1,7))
        P = np.ones((6,6), dtype=float)/36.0

    show_table(P, [f"x={x}" for x in x_vals], [f"y={y}" for y in y_vals], title="PMF conjunta p_{X,Y}")
    console.print(f"Suma total = {P.sum():.6f} (debe ser 1).")

    return P, x_vals, y_vals

def discrete_marginals(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pX = P.sum(axis=1)  # por filas
    pY = P.sum(axis=0)  # por columnas
    return pX, pY

def discrete_expectations(P: np.ndarray, x_vals: List[float], y_vals: List[float]):
    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    # Esperanzas
    EX = (P.sum(axis=1) * x_arr).sum()
    EY = (P.sum(axis=0) * y_arr).sum()
    # E[X^2], E[Y^2]
    EX2 = (P.sum(axis=1) * (x_arr**2)).sum()
    EY2 = (P.sum(axis=0) * (y_arr**2)).sum()
    VarX = EX2 - EX**2
    VarY = EY2 - EY**2
    # E[XY]
    XY = 0.0
    for i, xv in enumerate(x_vals):
        for j, yv in enumerate(y_vals):
            XY += xv*yv*P[i,j]
    Cov = XY - EX*EY
    # Corr
    if VarX <= 0 or VarY <= 0:
        Corr = float("nan")
    else:
        Corr = Cov / (VarX**0.5 * VarY**0.5)
    return EX, EY, VarX, VarY, Cov, Corr, XY

def check_independence_discrete(P: np.ndarray, pX: np.ndarray, pY: np.ndarray, tol=1e-9) -> bool:
    outer = np.outer(pX, pY)
    return np.allclose(P, outer, atol=tol)

def discrete_workflow():
    P, x_vals, y_vals = input_discrete()
    while True:
        console.rule("[bold]Operaciones discretas")
        console.print("  [1] Marginales p_X y p_Y")
        console.print("  [2] Condicionales p(Y|X=x0) y p(X|Y=y0)")
        console.print("  [3] Esperanzas/Var/Cov/Corr")
        console.print("  [4] Independencia (¿p_{X,Y} = p_X p_Y?)")
        console.print("  [5] Probabilidad de un evento (expresión simple)")
        console.print("  [6] Volver al menú principal")
        opt = IntPrompt.ask("Elige opción", choices=["1","2","3","4","5","6"], default="1")

        if opt == 1:
            pX, pY = discrete_marginals(P)
            # Mostrar
            table = Table(title="Marginales", box=box.MINIMAL_DOUBLE_HEAD)
            table.add_column("x", justify="right")
            table.add_column("p_X(x)", justify="right")
            for xv, px in zip(x_vals, pX):
                table.add_row(str(xv), f"{px:.6f}")
            console.print(table)

            table = Table(title="Marginales", box=box.MINIMAL_DOUBLE_HEAD)
            table.add_column("y", justify="right")
            table.add_column("p_Y(y)", justify="right")
            for yv, py in zip(y_vals, pY):
                table.add_row(str(yv), f"{py:.6f}")
            console.print(table)

        elif opt == 2:
            mode = IntPrompt.ask(" [1] p(Y|X=x0)  [2] p(X|Y=y0)", choices=["1","2"], default="1")
            if mode == 1:
                x0 = FloatPrompt.ask("Valor x0 (de la lista de X)")
                if x0 not in x_vals:
                    console.print("[red]x0 no está en los valores de X.[/red]")
                else:
                    i = x_vals.index(x0)
                    pX = P.sum(axis=1)
                    if pX[i] <= 0:
                        console.print("[red]p_X(x0)=0, condicional indefinida.[/red]")
                    else:
                        cond = P[i,:] / pX[i]
                        show_table(cond.reshape(1,-1), [f"p(Y|X={x0})"], [f"y={y}" for y in y_vals])
            else:
                y0 = FloatPrompt.ask("Valor y0 (de la lista de Y)")
                if y0 not in y_vals:
                    console.print("[red]y0 no está en los valores de Y.[/red]")
                else:
                    j = y_vals.index(y0)
                    pY = P.sum(axis=0)
                    if pY[j] <= 0:
                        console.print("[red]p_Y(y0)=0, condicional indefinida.[/red]")
                    else:
                        cond = P[:,j] / pY[j]
                        show_table(cond.reshape(-1,1), [f"x={x}" for x in x_vals], [f"p(X|Y={y0})"])

        elif opt == 3:
            EX, EY, VarX, VarY, Cov, Corr, EXY = discrete_expectations(P, x_vals, y_vals)
            info = Table(title="Momentos y dependencia", box=box.MINIMAL_DOUBLE_HEAD)
            info.add_column("Magnitud")
            info.add_column("Valor", justify="right")
            pairs = [
                ("E[X]", f"{EX:.6f}"),
                ("E[Y]", f"{EY:.6f}"),
                ("E[XY]", f"{EXY:.6f}"),
                ("Var(X)", f"{VarX:.6f}"),
                ("Var(Y)", f"{VarY:.6f}"),
                ("Cov(X,Y)", f"{Cov:.6f}"),
                ("Corr(X,Y)", f"{Corr:.6f}"),
            ]
            for k,v in pairs:
                info.add_row(k, v)
            console.print(info)

        elif opt == 4:
            pX, pY = discrete_marginals(P)
            indep = check_independence_discrete(P, pX, pY)
            console.print(Panel.fit("Independencia: [bold green]Sí[/bold green]" if indep else "Independencia: [bold red]No[/bold red]",
                                    title="p_{X,Y} = p_X p_Y ?", box=box.ROUNDED))

        elif opt == 5:
            console.print("Evento como combinación simple: ejemplo 'X>Y', 'X+Y=7', 'X<=3 and Y>=2'.", style="dim")
            expr = Prompt.ask("Expresión (usando X y Y)")
            # Evaluación segura: iterar sobre la grilla y sumar P[i,j] si cumple
            total = 0.0
            for i, xv in enumerate(x_vals):
                for j, yv in enumerate(y_vals):
                    # Crear variables locales
                    X, Y = xv, yv
                    try:
                        if eval(expr, {"__builtins__": {}}, {"X": X, "Y": Y}):
                            total += P[i,j]
                    except Exception:
                        console.print("[red]Expresión inválida.[/red]")
                        total = None
                        break
                if total is None:
                    break
            if total is not None:
                console.print(Panel.fit(f"P({expr}) = [bold]{total:.6f}[/bold]", title="Probabilidad de evento", box=box.ROUNDED))

        else:
            break

# =============================================================
# Caso CONTINUO
# =============================================================

def choose_continuous() -> Tuple[sp.Expr, Tuple[sp.Symbol, float, float], Tuple[sp.Symbol, float, float]]:
    console.rule("[bold]Caso continuo (PDF)")
    x, y = sp.symbols('x y', real=True)
    console.print("  [1] Ejemplo: f(x,y)=4xy en [0,1]×[0,1]")
    console.print("  [2] Ejemplo: f(x,y)=c en 0<y<x<1 (triangular)")
    console.print("  [3] Definir mi propia f(x,y) en un rectángulo [a,b]×[c,d]")
    opt = IntPrompt.ask("Opción", choices=["1","2","3"], default="1")

    if opt == 1:
        f = 4*x*y
        ax, bx = 0.0, 1.0
        ay, by = 0.0, 1.0
        return f, (x, ax, bx), (y, ay, by)

    elif opt == 2:
        # región triangular: 0<y<x<1 -> usamos función por partes: c si 0<y<x<1, 0 si no
        c = sp.symbols('c', positive=True)
        f = c  # valor constante en la región triangulo
        # Guardamos soporte como marcas; la integración la haremos con límites dependientes
        return sp.Piecewise((c, sp.And(sp.Gt(x,0), sp.Gt(y,0), sp.Lt(y,x), sp.Lt(x,1))), (0, True)), (x, 0.0, 1.0), (y, 0.0, 1.0)

    else:
        expr_str = Prompt.ask("Ingresa f(x,y) en sintaxis Sympy (ej: x**2 + y)")
        try:
            f = sp.sympify(expr_str, locals={"x": x, "y": y})
        except Exception as e:
            console.print(f"[red]No pude interpretar la expresión: {e}[/red]")
            f = x + y
        ax = FloatPrompt.ask("a (límite inferior de x)", default=0.0)
        bx = FloatPrompt.ask("b (límite superior de x)", default=1.0)
        ay = FloatPrompt.ask("c (límite inferior de y)", default=0.0)
        by = FloatPrompt.ask("d (límite superior de y)", default=1.0)
        return f, (x, ax, bx), (y, ay, by)

def continuous_normalize(f: sp.Expr, X: Tuple[sp.Symbol,float,float], Y: Tuple[sp.Symbol,float,float]) -> sp.Expr:
    x, ax, bx = X
    y, ay, by = Y
    # Si es la región triangular (Piecewise), integramos con límites dependientes
    if isinstance(f, sp.Piecewise):
        # 0<y<x<1 con densidad c
        c = list(f.free_symbols - {x, y})
        c = c[0] if c else sp.symbols('c')
        integral = sp.integrate(c, (x, 0, 1), (y, 0, x))
        c_val = sp.solve(sp.Eq(integral, 1), c)
        if c_val:
            f = f.subs({c: sp.simplify(c_val[0])})
        return f

    # Rectángulo regular
    integral = sp.integrate(f, (x, ax, bx), (y, ay, by))
    if integral == 0:
        return f
    if integral != 1:
        f = sp.simplify(f / integral)
    return f

def marginals_continuous(f: sp.Expr, X, Y):
    x, ax, bx = X
    y, ay, by = Y
    fX = sp.simplify(sp.integrate(f, (y, ay, by)))
    fY = sp.simplify(sp.integrate(f, (x, ax, bx)))
    return fX, fY

def conditional_continuous(f: sp.Expr, fX: sp.Expr, fY: sp.Expr):
    x, y = sp.symbols('x y', real=True)
    fy_given_x = sp.simplify(sp.Piecewise(
        (sp.simplify(sp.divide(f, sp.Max(fX, sp.Epsilon()))[0]), True)
    ))
    fx_given_y = sp.simplify(sp.Piecewise(
        (sp.simplify(sp.divide(f, sp.Max(fY, sp.Epsilon()))[0]), True)
    ))
    return fy_given_x, fx_given_y

def probability_rectangle(f: sp.Expr, X, Y, a: float, b: float, c: float, d: float):
    """P(a<=X<=b, c<=Y<=d) recortado al soporte"""
    x, ax, bx = X
    y, ay, by = Y
    aa, bb = max(ax, a), min(bx, b)
    cc, dd = max(ay, c), min(by, d)
    if aa >= bb or cc >= dd:
        return 0.0
    prob = sp.integrate(f, (x, aa, bb), (y, cc, dd))
    return sp.N(prob)

def check_independence_continuous(f: sp.Expr, fX: sp.Expr, fY: sp.Expr, X, Y) -> bool:
    """Verifica si f ≈ fX*fY por muestreo numérico en el soporte."""
    x, ax, bx = X
    y, ay, by = Y
    fx_y = sp.lambdify((x,y), f, "numpy")
    fx = sp.lambdify((x,), fX, "numpy")
    fy = sp.lambdify((y,), fY, "numpy")
    xs = np.linspace(ax, bx, 7)
    ys = np.linspace(ay, by, 7)
    for xv in xs:
        for yv in ys:
            lhs = float(fx_y(xv, yv))
            rhs = float(fx(xv) * fy(yv))
            if not np.isfinite(lhs) or not np.isfinite(rhs):
                continue
            if abs(lhs - rhs) > 1e-6 * max(1.0, abs(rhs)):
                return False
    return True

def expectations_continuous(f: sp.Expr, X, Y):
    x, ax, bx = X
    y, ay, by = Y
    EX = sp.integrate(sp.integrate(x*f, (y, ay, by)), (x, ax, bx))
    EY = sp.integrate(sp.integrate(y*f, (x, ax, bx)), (y, ay, by))
    EX2 = sp.integrate(sp.integrate((x**2)*f, (y, ay, by)), (x, ax, bx))
    EY2 = sp.integrate(sp.integrate((y**2)*f, (x, ax, bx)), (y, ay, by))
    EXY = sp.integrate(sp.integrate((x*y)*f, (y, ay, by)), (x, ax, bx))
    VarX = sp.simplify(EX2 - EX**2)
    VarY = sp.simplify(EY2 - EY**2)
    Cov = sp.simplify(EXY - EX*EY)
    Corr = sp.simplify(Cov / sp.sqrt(VarX*VarY)) if VarX != 0 and VarY != 0 else sp.nan
    return sp.simplify(EX), sp.simplify(EY), sp.simplify(VarX), sp.simplify(VarY), sp.simplify(Cov), sp.simplify(Corr), sp.simplify(EXY)

def continuous_workflow():
    f, X, Y = choose_continuous()
    console.print("\n[bold]Densidad propuesta (antes de normalizar):[/bold]")
    pretty_expr(f, prefix="f(x,y) = ")
    f = continuous_normalize(f, X, Y)
    console.print("[bold]Densidad normalizada:[/bold]")
    pretty_expr(f, prefix="f(x,y) = ")

    while True:
        console.rule("[bold]Operaciones continuas")
        console.print("  [1] Marginales f_X y f_Y")
        console.print("  [2] Condicionales f(Y|X=x) y f(X|Y=y) (forma simbólica)")
        console.print("  [3] E[X], E[Y], Var, Cov, Corr")
        console.print("  [4] P(a≤X≤b, c≤Y≤d)")
        console.print("  [5] Independencia (¿f = f_X f_Y?)")
        console.print("  [6] Volver al menú principal")
        opt = IntPrompt.ask("Elige opción", choices=["1","2","3","4","5","6"], default="1")

        if opt == 1:
            fX, fY = marginals_continuous(f, X, Y)
            console.print("[bold]f_X(x):[/bold]")
            pretty_expr(fX, prefix="f_X(x) = ")
            console.print("[bold]f_Y(y):[/bold]")
            pretty_expr(fY, prefix="f_Y(y) = ")

        elif opt == 2:
            fX, fY = marginals_continuous(f, X, Y)
            fyx, fxy = conditional_continuous(f, fX, fY)
            console.print("[bold]f(Y|X=x):[/bold]")
            pretty_expr(fyx, prefix="f_{Y|X}(y|x) = ")
            console.print("[bold]f(X|Y=y):[/bold]")
            pretty_expr(fxy, prefix="f_{X|Y}(x|y) = ")

        elif opt == 3:
            EX, EY, VarX, VarY, Cov, Corr, EXY = expectations_continuous(f, X, Y)
            table = Table(title="Momentos (forma exacta)", box=box.MINIMAL_DOUBLE_HEAD)
            for k, v in [
                ("E[X]", EX), ("E[Y]", EY), ("E[XY]", EXY),
                ("Var(X)", VarX), ("Var(Y)", VarY),
                ("Cov(X,Y)", Cov), ("Corr(X,Y)", Corr)
            ]:
                table.add_row(k, sp.sstr(v))
            console.print(table)

        elif opt == 4:
            a = FloatPrompt.ask("a (límite inferior de X)", default=float(X[1]))
            b = FloatPrompt.ask("b (límite superior de X)", default=float(X[2]))
            c = FloatPrompt.ask("c (límite inferior de Y)", default=float(Y[1]))
            d = FloatPrompt.ask("d (límite superior de Y)", default=float(Y[2]))
            prob = probability_rectangle(f, X, Y, a, b, c, d)
            console.print(Panel.fit(f"P({a}≤X≤{b}, {c}≤Y≤{d}) = [bold]{prob}[/bold]",
                                    title="Probabilidad rectangular", box=box.ROUNDED))

        elif opt == 5:
            fX, fY = marginals_continuous(f, X, Y)
            indep = check_independence_continuous(f, fX, fY, X, Y)
            console.print(Panel.fit("Independencia: [bold green]Sí[/bold green]" if indep else "Independencia: [bold red]No[/bold red]",
                                    title="f = f_X f_Y ?", box=box.ROUNDED))
        else:
            break

# =============================================================
# Menú principal
# =============================================================

def main():
    banner()
    while True:
        console.rule("[bold]Menú principal")
        console.print("  [1] Caso discreto (PMF)")
        console.print("  [2] Caso continuo (PDF)")
        console.print("  [3] Salir")
        opt = IntPrompt.ask("Elige opción", choices=["1","2","3"], default="1")
        if opt == 1:
            discrete_workflow()
        elif opt == 2:
            continuous_workflow()
        else:
            console.print("\n¡Hasta luego!")
            break
        pause()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Interrumpido por el usuario.[/red]")
