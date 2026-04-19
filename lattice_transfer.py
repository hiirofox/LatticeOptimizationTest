#!/usr/bin/env python3
"""
Symbolic transfer function for a damped nested lattice structure.

Each section is represented as

    H_i = (k_i + d_i * Hd_i * H_{i+1})
          --------------------------------
          (1   + k_i * d_i * Hd_i * H_{i+1})

where Hd_i stands for the delay block z^(-n_i).

The default terminal is H_{n+1} = 1, so one section becomes

    H_1 = (k_1 + d_1 * Hd_1) / (1 + k_1 * d_1 * Hd_1)

The outer feedback is

    lattice_in  = x + d_out * lattice_out
    lattice_out = H_1 * lattice_in

so the closed-loop transfer function from x to lattice_out is

    H_total = H_1 / (1 - d_out * H_1)

Run:

    python lattice_transfer.py 3
    python lattice_transfer.py 3 --open-loop
    python lattice_transfer.py 3 --latex
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class LatticeSymbols:
    k: tuple[sp.Symbol, ...]
    d: tuple[sp.Symbol, ...]
    Hd: tuple[sp.Symbol, ...]
    d_out: sp.Symbol


def make_symbols(num_sections: int) -> LatticeSymbols:
    """Create k_i, d_i, Hd_i, and d_out symbols."""
    if num_sections < 1:
        raise ValueError("num_sections must be >= 1")

    return LatticeSymbols(
        k=sp.symbols(f"k_1:{num_sections + 1}"),
        d=sp.symbols(f"d_1:{num_sections + 1}"),
        Hd=sp.symbols(f"Hd_1:{num_sections + 1}"),
        d_out=sp.Symbol("d_out"),
    )


def lattice_open_loop(num_sections: int) -> tuple[sp.Expr, sp.Expr, LatticeSymbols]:
    """
    Return the expanded numerator and denominator of H_1.

    The recursion is evaluated as numerator/denominator pairs to keep the
    result directly in rational polynomial form:

        H_i = N_i / D_i

        N_i = k_i * D_{i+1} + d_i * Hd_i * N_{i+1}
        D_i =       D_{i+1} + k_i * d_i * Hd_i * N_{i+1}
    """
    symbols = make_symbols(num_sections)

    numerator = sp.Integer(1)
    denominator = sp.Integer(1)

    for i in reversed(range(num_sections)):
        section_delay = symbols.d[i] * symbols.Hd[i]
        numerator, denominator = (
            sp.expand(symbols.k[i] * denominator + section_delay * numerator),
            sp.expand(denominator + symbols.k[i] * section_delay * numerator),
        )

    return numerator, denominator, symbols


def lattice_closed_loop(num_sections: int) -> tuple[sp.Expr, sp.Expr, LatticeSymbols]:
    """
    Return the expanded numerator and denominator of lattice_out / x.

    If H_1 = N / D, then with positive feedback d_out:

        H_total = (N / D) / (1 - d_out * N / D)
                = N / (D - d_out * N)
    """
    numerator, denominator, symbols = lattice_open_loop(num_sections)
    return (
        sp.expand(numerator),
        sp.expand(denominator - symbols.d_out * numerator),
        symbols,
    )


def transfer_expression(numerator: sp.Expr, denominator: sp.Expr) -> sp.Expr:
    """Build a SymPy rational expression from numerator and denominator."""
    return numerator / denominator


def print_result(num_sections: int, closed_loop: bool, latex: bool) -> None:
    if closed_loop:
        numerator, denominator, symbols = lattice_closed_loop(num_sections)
        title = "Closed-loop transfer function lattice_out / x"
    else:
        numerator, denominator, symbols = lattice_open_loop(num_sections)
        title = "Open-loop lattice transfer function H_1"

    expression = transfer_expression(numerator, denominator)

    print(title)
    print("=" * len(title))
    print(f"num_sections = {num_sections}")
    print(f"k symbols    = {symbols.k}")
    print(f"d symbols    = {symbols.d}")
    print(f"Hd symbols   = {symbols.Hd}")
    print(f"d_out symbol = {symbols.d_out}")
    print()
    print("Numerator polynomial:")
    print(sp.expand(numerator))
    print()
    print("Denominator polynomial:")
    print(sp.expand(denominator))
    print()
    print("Transfer function:")
    print(expression)

    if latex:
        print()
        print("LaTeX numerator:")
        print(sp.latex(sp.expand(numerator)))
        print()
        print("LaTeX denominator:")
        print(sp.latex(sp.expand(denominator)))
        print()
        print("LaTeX transfer function:")
        print(sp.latex(expression))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate numerator/denominator polynomials for the nested damped lattice."
    )
    parser.add_argument("num_sections", type=int, help="number of lattice sections")
    parser.add_argument(
        "--open-loop",
        action="store_true",
        help="print H_1 only, without the outer d_out feedback",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="also print LaTeX output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_result(
        num_sections=args.num_sections,
        closed_loop=not args.open_loop,
        latex=args.latex,
    )


if __name__ == "__main__":
    main()
