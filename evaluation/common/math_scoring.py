"""
Scoring utilities for Level 0 evaluation.
Shared mathematical answer-equivalence utilities.
Provides is_equal() with latex2sympy support and find_math_answer().
"""
import re
from typing import Optional


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def delete_extra_zero(n: str) -> str:
    try:
        n_f = float(n)
        if n_f == int(n_f):
            return str(int(n_f))
        return str(n_f).rstrip('0').rstrip('.')
    except Exception:
        return n


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if not substr or substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + post_substr
    return new_str


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split or split[0] != "{":
            a = split[0] if split else ""
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("\\%", "")
    string = string.replace(" ", "")

    string = _fix_sqrt(string)
    string = _fix_fracs(string)

    if not string.startswith("\\") and is_number(string):
        string = delete_extra_zero(string)

    if len(string) > 0 and string[0] == ".":
        string = "0" + string
    if len(string) > 1 and string[-1] == ".":
        string = string[:-1]

    return string


def find_math_answer(s: str) -> str:
    if '\\boxed' in s:
        ans = s.split('\\boxed')[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    a += c
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
            return _strip_string(a)
        else:
            return _strip_string(ans.split('$')[0])
    elif '\\text{' in s:
        return _strip_string(s.split('\\text{')[-1].split('}')[0])
    else:
        return _strip_string(s)


def is_equal(asw: str, gt_asw: str) -> bool:
    # Normalise case first so "A" == "a" (matches original eval behaviour)
    asw = asw.lower()
    gt_asw = gt_asw.lower()

    if not asw.replace(" ", "") or not gt_asw.replace(" ", ""):
        return False

    if asw == gt_asw:
        return True

    asw_stripped = _strip_string(asw)
    gt_stripped = _strip_string(gt_asw)

    if asw_stripped == gt_stripped:
        return True

    if is_number(asw_stripped) and is_number(gt_stripped):
        if abs(float(asw_stripped) - float(gt_stripped)) < 1e-6:
            return True

    try:
        from latex2sympy2 import latex2sympy
        asw_sym = latex2sympy(asw_stripped)
        gt_sym = latex2sympy(gt_stripped)
        if abs(float(asw_sym) - float(gt_sym)) < 1e-6:
            return True
    except Exception:
        pass

    return False
