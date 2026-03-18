#!/usr/bin/env python3
"""Persistent MLIP model server and thin client.

The server loads the MLIP model once and keeps it resident in memory,
accepting evaluation requests over a Unix domain socket.  The client
is a lightweight function called by the Gaussian/ORCA runner each time
the external program is invoked, avoiding repeated model loading.

Wire protocol
-------------
Each message is a length-prefixed JSON blob:

    [4-byte big-endian uint32: payload length] [UTF-8 JSON payload]

Actions
-------
- ``evaluate`` : run evaluator.evaluate() and return results
- ``ping``     : health check, returns ``{"status": "ok"}``
- ``shutdown`` : graceful server stop
"""

from __future__ import absolute_import, division, print_function

import hashlib
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np


class ServerError(RuntimeError):
    """Raised when server communication fails."""


def _pid_is_alive(pid):
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# Wire protocol helpers
# ---------------------------------------------------------------------------

_HEADER_FMT = "!I"  # 4-byte big-endian unsigned int
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MSG_SIZE = 256 * 1024 * 1024  # 256 MB safety cap


def _recv_exact(sock, nbytes):
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _send_msg(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    header = struct.pack(_HEADER_FMT, len(data))
    sock.sendall(header + data)


def _recv_msg(sock):
    header = _recv_exact(sock, _HEADER_SIZE)
    if header is None:
        return None
    (length,) = struct.unpack(_HEADER_FMT, header)
    if length > _MAX_MSG_SIZE:
        raise ServerError("Message too large: {} bytes".format(length))
    data = _recv_exact(sock, length)
    if data is None:
        raise ServerError("Connection closed mid-message")
    return json.loads(data.decode("utf-8"))


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class MLIPServer(object):
    """Single-threaded Unix domain socket server wrapping an evaluator."""

    def __init__(self, evaluator, socket_path, idle_timeout=600, parent_pid=None):
        self.evaluator = evaluator
        self.socket_path = os.path.abspath(socket_path)
        self.idle_timeout = float(idle_timeout)
        self.parent_pid = int(parent_pid) if parent_pid is not None else None
        self._running = False
        self._last_activity = time.time()

    def serve_forever(self):
        if os.path.exists(self.socket_path):
            if server_is_alive(self.socket_path, timeout=2.0):
                print(
                    "[mlip-server] Another server is already running at {}".format(
                        self.socket_path
                    ),
                    file=sys.stderr,
                )
                return
            os.unlink(self.socket_path)

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        srv.listen(5)
        srv.settimeout(1.0)

        self._running = True
        self._last_activity = time.time()

        prev_sigint = None
        prev_sigterm = None
        try:
            prev_sigint = signal.getsignal(signal.SIGINT)
            prev_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except (ValueError, OSError):
            pass  # Not in main thread; skip signal handling

        print(
            "[mlip-server] Listening on {} (idle_timeout={}s, evaluator={}, parent_pid={})".format(
                self.socket_path,
                int(self.idle_timeout),
                self.evaluator.__class__.__name__,
                self.parent_pid if self.parent_pid is not None else "none",
            ),
            file=sys.stderr,
            flush=True,
        )

        try:
            while self._running:
                if self.parent_pid is not None and not _pid_is_alive(self.parent_pid):
                    print(
                        "[mlip-server] Parent process {} exited, shutting down.".format(
                            self.parent_pid
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    break

                # Idle timeout check
                if time.time() - self._last_activity > self.idle_timeout:
                    print(
                        "[mlip-server] Idle timeout reached, shutting down.",
                        file=sys.stderr,
                        flush=True,
                    )
                    break

                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue

                try:
                    self._handle_connection(conn)
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                finally:
                    conn.close()
        finally:
            srv.close()
            if os.path.exists(self.socket_path):
                try:
                    os.unlink(self.socket_path)
                except OSError:
                    pass
            if prev_sigint is not None:
                try:
                    signal.signal(signal.SIGINT, prev_sigint)
                    signal.signal(signal.SIGTERM, prev_sigterm)
                except (ValueError, OSError):
                    pass
            print("[mlip-server] Shut down.", file=sys.stderr, flush=True)

    def _handle_signal(self, signum, frame):
        print(
            "\n[mlip-server] Signal {} received, shutting down...".format(signum),
            file=sys.stderr,
            flush=True,
        )
        self._running = False

    def _handle_connection(self, conn):
        conn.settimeout(600.0)
        request = _recv_msg(conn)
        if request is None:
            return

        self._last_activity = time.time()
        action = request.get("action", "")

        if action == "ping":
            _send_msg(conn, {"status": "ok", "message": "pong"})
            return

        if action == "shutdown":
            _send_msg(conn, {"status": "ok", "message": "shutting down"})
            self._running = False
            return

        if action == "evaluate":
            response = self._do_evaluate(request)
            _send_msg(conn, response)
            return

        _send_msg(
            conn, {"status": "error", "message": "Unknown action: {}".format(action)}
        )

    def _do_evaluate(self, request):
        try:
            symbols = request["symbols"]
            coords_ang = np.asarray(request["coords_ang"], dtype=np.float64)
            charge = int(request["charge"])
            multiplicity = int(request["multiplicity"])
            need_forces = bool(request.get("need_forces", True))
            need_hessian = bool(request.get("need_hessian", False))
            hessian_mode = str(request.get("hessian_mode", "Analytical"))
            hessian_step = float(request.get("hessian_step", 1.0e-3))

            energy_ev, forces_ev_ang, hess_ev_ang2 = self.evaluator.evaluate(
                symbols=symbols,
                coords_ang=coords_ang,
                charge=charge,
                multiplicity=multiplicity,
                need_forces=need_forces,
                need_hessian=need_hessian,
                hessian_mode=hessian_mode,
                hessian_step=hessian_step,
            )

            return {
                "status": "ok",
                "energy_ev": float(energy_ev),
                "forces_ev_ang": forces_ev_ang.tolist()
                if forces_ev_ang is not None
                else None,
                "hessian_ev_ang2": hess_ev_ang2.tolist()
                if hess_ev_ang2 is not None
                else None,
            }
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------


def server_is_alive(socket_path, timeout=5.0):
    """Return True if a server is responding at *socket_path*."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(os.path.abspath(socket_path))
        _send_msg(sock, {"action": "ping"})
        resp = _recv_msg(sock)
        return resp is not None and resp.get("status") == "ok"
    except Exception:
        return False
    finally:
        sock.close()


def send_shutdown(socket_path, timeout=10.0):
    """Send a shutdown command to the server."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(os.path.abspath(socket_path))
        _send_msg(sock, {"action": "shutdown"})
        return _recv_msg(sock)
    finally:
        sock.close()


def client_evaluate(
    socket_path,
    symbols,
    coords_ang,
    charge,
    multiplicity,
    need_forces,
    need_hessian,
    hessian_mode,
    hessian_step,
    timeout=600.0,
):
    """Connect to a running MLIPServer and request an evaluation.

    Returns ``(energy_ev, forces_ev_ang_or_None, hessian_ev_ang2_or_None)``.
    """
    socket_path = os.path.abspath(socket_path)
    if not os.path.exists(socket_path):
        raise ServerError(
            "Server socket not found: {}. Is the server running?".format(socket_path)
        )

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(socket_path)
        request = {
            "action": "evaluate",
            "symbols": list(symbols),
            "coords_ang": np.asarray(coords_ang, dtype=np.float64).tolist(),
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "need_forces": bool(need_forces),
            "need_hessian": bool(need_hessian),
            "hessian_mode": str(hessian_mode),
            "hessian_step": float(hessian_step),
        }
        _send_msg(sock, request)
        response = _recv_msg(sock)
    finally:
        sock.close()

    if response is None:
        raise ServerError("No response from server")
    if response.get("status") != "ok":
        raise ServerError("Server error: {}".format(response.get("message", "unknown")))

    energy_ev = float(response["energy_ev"])
    forces_ev_ang = (
        np.asarray(response["forces_ev_ang"], dtype=np.float64)
        if response.get("forces_ev_ang") is not None
        else None
    )
    hess_ev_ang2 = (
        np.asarray(response["hessian_ev_ang2"], dtype=np.float64)
        if response.get("hessian_ev_ang2") is not None
        else None
    )
    return energy_ev, forces_ev_ang, hess_ev_ang2


# ---------------------------------------------------------------------------
# Auto-start helpers
# ---------------------------------------------------------------------------


def auto_server_socket(args, parent_pid=None):
    """Compute a deterministic socket path from evaluator arguments."""
    key_parts = [str(getattr(args, "model", "default"))]
    if hasattr(args, "device"):
        key_parts.append(str(args.device))
    if hasattr(args, "task"):
        key_parts.append(str(args.task))
    if hasattr(args, "precision"):
        key_parts.append(str(args.precision))
    if parent_pid is None:
        parent_pid = os.getppid()
    key_parts.append("ppid={}".format(int(parent_pid)))

    key = "_".join(key_parts)
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    uid = os.getuid()
    return os.path.join(
        tempfile.gettempdir(),
        "mlip_server_{uid}_{hash}.sock".format(uid=uid, hash=h),
    )


def _build_serve_argv(
    executable, custom_args, socket_path, idle_timeout, parent_pid=None
):
    """Build the argv for spawning the server subprocess."""
    cmd = [sys.executable, executable]
    cmd.extend(custom_args)
    cmd.extend(["--serve", "--server-socket", socket_path])
    if idle_timeout is not None:
        cmd.extend(["--server-idle-timeout", str(int(idle_timeout))])
    if parent_pid is not None:
        cmd.extend(["--server-parent-pid", str(int(parent_pid))])
    return cmd


def ensure_server(
    executable,
    custom_args,
    socket_path,
    idle_timeout=600,
    parent_pid=None,
):
    """Ensure a server is running at *socket_path*.

    If no server is alive, spawn one as a background subprocess and wait
    until it becomes ready.

    Returns True if the server is ready, False otherwise.
    """
    if server_is_alive(socket_path, timeout=2.0):
        return True

    # Clean up stale socket file
    if os.path.exists(socket_path):
        try:
            os.unlink(socket_path)
        except OSError:
            pass

    cmd = _build_serve_argv(
        executable=executable,
        custom_args=custom_args,
        socket_path=socket_path,
        idle_timeout=idle_timeout,
        parent_pid=parent_pid,
    )
    print(
        "[mlip-client] Starting server: {}".format(" ".join(cmd)),
        file=sys.stderr,
        flush=True,
    )

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
            start_new_session=True,
        )
    except Exception as exc:
        print(
            "[mlip-client] WARNING: Failed to start server: {}".format(exc),
            file=sys.stderr,
            flush=True,
        )
        return False

    startup_timeout = 300
    waited = 0
    while waited < startup_timeout:
        if proc.poll() is not None:
            print(
                "[mlip-client] WARNING: Server process exited unexpectedly (code={}).".format(
                    proc.returncode
                ),
                file=sys.stderr,
                flush=True,
            )
            return False

        if server_is_alive(socket_path, timeout=1.0):
            print("[mlip-client] Server is ready.", file=sys.stderr, flush=True)
            return True

        time.sleep(1)
        waited += 1

    print(
        "[mlip-client] WARNING: Server did not become ready within {}s.".format(
            startup_timeout
        ),
        file=sys.stderr,
        flush=True,
    )
    return False
