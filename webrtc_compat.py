"""Compatibilidad de WebRTC para cierres de transporte en Python 3.14."""

from __future__ import annotations

import sys
from typing import Any


def instalar_guardia_reintento_aioice(*, forzar: bool = False) -> bool:
    """
    Evita que un temporizador STUN escriba sobre un transporte ya cerrado.

    aioice 0.10.2 puede conservar brevemente un reintento programado después
    de que asyncio cierre el socket UDP. En Python 3.14 esa carrera termina en
    un AttributeError dentro del callback del event loop. La transacción debe
    finalizar como timeout, que es la semántica normal de un STUN inaccesible.

    Devuelve True únicamente cuando instala la protección.
    """
    if sys.version_info < (3, 14) and not forzar:
        return False

    try:
        from aioice import stun
    except ImportError:
        return False

    nombre_metodo = "_Transaction__retry"
    metodo_original = getattr(stun.Transaction, nombre_metodo, None)
    if metodo_original is None or getattr(
        metodo_original, "_guardia_transporte_cerrado", False
    ):
        return False

    def reintento_seguro(self: Any) -> None:
        futuro = getattr(self, "_Transaction__future")
        if futuro.done():
            return

        try:
            metodo_original(self)
        except AttributeError as error:
            mensaje = str(error)
            transporte_cerrado = (
                "'NoneType' object has no attribute 'sendto'" in mensaje
                or "'NoneType' object has no attribute "
                "'call_exception_handler'" in mensaje
            )
            if not transporte_cerrado:
                raise
            if not futuro.done():
                futuro.set_exception(stun.TransactionTimeout())

    reintento_seguro._guardia_transporte_cerrado = True  # type: ignore[attr-defined]
    setattr(stun.Transaction, nombre_metodo, reintento_seguro)
    return True
