"""Pruebas para la protección de cierre STUN."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from aioice.stun import Transaction, TransactionTimeout

from webrtc_compat import instalar_guardia_reintento_aioice


class ProtocoloConTransporteCerrado:
    """Simula el transporte UDP cerrado observado en Python 3.14."""

    def send_stun(self, message: Any, addr: tuple[str, int]) -> None:
        raise AttributeError(
            "'NoneType' object has no attribute 'call_exception_handler'"
        )


class GuardiaAioiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_convierte_cierre_de_transporte_en_timeout(self) -> None:
        self.assertTrue(instalar_guardia_reintento_aioice(forzar=True))
        transaccion = Transaction(
            request=object(),  # type: ignore[arg-type]
            addr=("127.0.0.1", 19302),
            protocol=ProtocoloConTransporteCerrado(),
        )

        with self.assertRaises(TransactionTimeout):
            await asyncio.wait_for(transaccion.run(), timeout=1)
        self.assertFalse(instalar_guardia_reintento_aioice(forzar=True))


if __name__ == "__main__":
    unittest.main()
