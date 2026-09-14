"""Backend registry.

Every ``*.py`` in this folder that is not ``base.py`` and does not start with ``_`` is
imported at load time and scanned for ``SliderBackend`` subclasses. Dropping a new file
in here is the entire installation procedure for a new base model -- nothing else in the
package needs editing, and a backend that fails to import is reported and skipped rather
than taking the whole node pack down with it.
"""

import importlib
import inspect
import logging
import pkgutil

from .base import SliderBackend, modules_on_device

LOG = "[ZeroTrainSlider]"

_BACKENDS = {}
_IMPORT_ERRORS = {}


def _discover():
    if _BACKENDS or _IMPORT_ERRORS:
        return
    package = __name__
    for info in pkgutil.iter_modules(__path__):
        name = info.name
        if name.startswith("_") or name == "base":
            continue
        try:
            module = importlib.import_module("{}.{}".format(package, name))
        except Exception as exc:  # noqa: BLE001 - one bad backend must not break the rest
            _IMPORT_ERRORS[name] = repr(exc)
            logging.exception("%s Backend %r failed to import; skipping.", LOG, name)
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, SliderBackend) or obj is SliderBackend:
                continue
            if not obj.key:
                continue
            if obj.key in _BACKENDS and _BACKENDS[obj.key] is not obj:
                logging.warning(
                    "%s Two backends claim key %r; keeping %s.",
                    LOG, obj.key, _BACKENDS[obj.key].__name__,
                )
                continue
            _BACKENDS[obj.key] = obj


def all_backends():
    """{key: backend class}, discovery-ordered."""
    _discover()
    return dict(_BACKENDS)


def import_errors():
    """{module name: repr of the exception}, for backends that would not import."""
    _discover()
    return dict(_IMPORT_ERRORS)


def get_backend(key):
    _discover()
    try:
        return _BACKENDS[key]
    except KeyError:
        raise ValueError(
            "{} Unknown backend {!r}. Installed: {}.".format(
                LOG, key, ", ".join(sorted(_BACKENDS)) or "(none)"
            )
        )


def detect(model_patcher):
    """Pick the backend that recognises this MODEL, or raise a legible error."""
    _discover()
    for backend in _BACKENDS.values():
        try:
            if backend.matches(model_patcher):
                return backend
        except Exception:  # noqa: BLE001 - a broken matches() must not hide the others
            logging.exception("%s Backend %r raised in matches().", LOG, backend.key)
    installed = ", ".join(
        "{} ({})".format(b.display_name, b.key) for b in _BACKENDS.values()
    ) or "(none)"
    extra = ""
    if _IMPORT_ERRORS:
        extra = "\nBackends that failed to import: {}.".format(
            ", ".join(sorted(_IMPORT_ERRORS))
        )
    raise ValueError(
        "{} No installed backend recognises this MODEL (it is a {}).\n"
        "Installed backends: {}.{}\n"
        "Adding support is one file in zt_backends/ -- see ADDING_A_MODEL.md.".format(
            LOG, type(model_patcher.model).__name__, installed, extra
        )
    )


def all_site_presets():
    """Union of every backend's edit-site preset names, for the static widget list."""
    _discover()
    names = []
    for backend in _BACKENDS.values():
        for preset in backend.site_presets:
            if preset not in names:
                names.append(preset)
    return names


__all__ = [
    "SliderBackend",
    "modules_on_device",
    "all_backends",
    "all_site_presets",
    "detect",
    "get_backend",
    "import_errors",
]
