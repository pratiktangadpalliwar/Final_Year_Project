import pytest

from server.app.ws_hub import WsHub


class FakeSocket:
    def __init__(self):
        self.sent: list[dict] = []
        self.closed = False

    async def send_json(self, payload):
        if self.closed:
            raise RuntimeError("closed")
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_broadcast_reaches_all_subscribers():
    hub = WsHub()
    a, b = FakeSocket(), FakeSocket()
    hub.add(a)
    hub.add(b)
    await hub.broadcast({"type": "round_started", "round": 1})
    assert a.sent == [{"type": "round_started", "round": 1}]
    assert b.sent == [{"type": "round_started", "round": 1}]


@pytest.mark.asyncio
async def test_broken_socket_is_dropped_silently():
    hub = WsHub()
    good = FakeSocket()
    bad = FakeSocket(); bad.closed = True
    hub.add(good); hub.add(bad)
    await hub.broadcast({"type": "x"})
    assert hub.size() == 1  # bad dropped
    assert good.sent == [{"type": "x"}]


@pytest.mark.asyncio
async def test_remove():
    hub = WsHub()
    s = FakeSocket()
    hub.add(s); hub.remove(s)
    assert hub.size() == 0
