from __future__ import annotations

import importlib.util
from pathlib import Path
import threading
from urllib.request import urlopen

import numpy as np
from openpi_client import msgpack_numpy
import websockets.sync.client
from websockets.sync.server import serve


_FAKE_MACHINE_A_PATH = Path(__file__).resolve().parents[1] / "launch" / "fake_machine_a.py"
_SPEC = importlib.util.spec_from_file_location("fake_machine_a", _FAKE_MACHINE_A_PATH)
assert _SPEC is not None and _SPEC.loader is not None
fake_machine_a = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fake_machine_a)


def test_healthz_and_websocket_payload_are_compatible() -> None:
    packer = msgpack_numpy.Packer()
    with serve(
        fake_machine_a.handler,
        "127.0.0.1",
        0,
        max_size=None,
        compression=None,
        process_request=fake_machine_a.healthz,
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2.0) as response:
            assert response.status == 200
            assert response.read() == b"OK\n"
        with websockets.sync.client.connect(f"ws://127.0.0.1:{port}", compression=None, open_timeout=2.0) as ws:
            metadata = msgpack_numpy.unpackb(ws.recv(timeout=2.0))
            assert metadata["server"] == "fake-machine-a"
            assert metadata["supports_batch"] is True
            ws.send(packer.pack({"state": np.arange(7, dtype=np.float32)}))
            payload = msgpack_numpy.unpackb(ws.recv(timeout=2.0))
            ws.send(packer.pack({"batch": [{"state": np.zeros(7, dtype=np.float32)}, {"state": np.ones(7, dtype=np.float32)}]}))
            batch_payload = msgpack_numpy.unpackb(ws.recv(timeout=2.0))
        assert payload["z_rl"].shape == (fake_machine_a.Z_DIM,)
        assert payload["proprio"].shape == (fake_machine_a.PROPRIO_DIM,)
        assert payload["ref_chunk"].shape == (fake_machine_a.CHUNK_LEN, fake_machine_a.ACTION_DIM)
        np.testing.assert_allclose(payload["proprio"], np.arange(7, dtype=np.float32))
        assert len(batch_payload["batch_results"]) == 2
        np.testing.assert_allclose(batch_payload["batch_results"][0]["proprio"], np.zeros(7, dtype=np.float32))
        np.testing.assert_allclose(batch_payload["batch_results"][1]["proprio"], np.ones(7, dtype=np.float32))
        server.shutdown()
        thread.join(timeout=2.0)
