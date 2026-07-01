#!/usr/bin/env python3
"""Kiem tra Paddle GPU trong env ste.

paddle.utils.run_check() (Paddle 2.6) loi voi numpy 2.x:
  np.array(tensor, copy=True) -> TypeError __array__(copy=True)

Script nay kiem tra tinh toan GPU that (du pipeline OCR can),
chi goi run_check() khi numpy < 2.
"""
from __future__ import annotations

import sys


def main() -> int:
    import numpy as np
    import paddle

    print(f"numpy {np.__version__}")
    paddle.set_device("gpu:0")
    x = paddle.randn([4, 4])
    y = paddle.matmul(x, x)
    z = y.numpy()
    if z.shape != (4, 4):
        print(f"[LOI] Paddle GPU shape sai: {z.shape}", file=sys.stderr)
        return 1
    print("[OK] Paddle GPU — tinh toan tensor tren GPU thanh cong.")

    if int(np.__version__.split(".", maxsplit=1)[0]) < 2:
        paddle.utils.run_check()
    else:
        print(
            "[INFO] Bo qua paddle.utils.run_check() (khong tuong thich numpy 2.x).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
