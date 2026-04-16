import asyncio
import sys
import types
from types import SimpleNamespace

sys.modules.setdefault(
    "http_server",
    types.SimpleNamespace(start_http_server=lambda *args, **kwargs: None),
)

from main import MarketPlugin


class FakeProvider:
    def __init__(self):
        self.registered = set()

    def register_crypto_symbols(self, symbols):
        self.registered.update(symbols)

    def is_crypto(self, symbol: str) -> bool:
        return symbol.upper() in self.registered or symbol.upper() in {"BTC", "ETH", "SOL"}


class FakePostgres:
    def __init__(self, caches=None):
        self.caches = caches or {}
        self.saved = {}

    async def load_caches(self):
        return dict(self.caches)

    async def save_caches(self, caches):
        self.saved.update(caches)
        self.caches.update(caches)


class FakeHistory:
    def __init__(self, symbol, domain):
        self.symbol = symbol
        self.domain = domain


async def main():
    plugin = MarketPlugin()
    plugin._provider = FakeProvider()

    cached_state = {
        plugin.STATE_CACHE_KEY: {
            "crypto_symbols": ["btc", "eth"],
            "stock_symbols": ["aapl", "msft"],
            "crypto_offset": 7,
            "stock_offset": 3,
            "last_discovery_ts": 123.0,
            "last_fetch_ts": 124.0,
            "last_successful_fetch_ts": 125.0,
            "total_observations_emitted": 42,
        }
    }
    postgres = FakePostgres(caches=cached_state)
    core = SimpleNamespace(
        postgres=postgres,
        prediction_engine=SimpleNamespace(symbols={}),
        cognitive_system=SimpleNamespace(_price_history={}, _meta_domains={}),
    )

    await plugin.restore_runtime_state(core)
    exported = plugin.export_runtime_state()
    assert exported["crypto_symbols"] == ["BTC", "ETH"], exported
    assert exported["stock_symbols"] == ["AAPL", "MSFT"], exported
    assert plugin._provider.registered == {"BTC", "ETH"}, plugin._provider.registered
    assert plugin._crypto_offset == 1, plugin._crypto_offset
    assert plugin._stock_offset == 1, plugin._stock_offset

    plugin2 = MarketPlugin()
    plugin2._provider = FakeProvider()
    postgres2 = FakePostgres(caches={})
    core2 = SimpleNamespace(
        postgres=postgres2,
        prediction_engine=SimpleNamespace(symbols={
            "BTC": FakeHistory("BTC", "crypto:BTC"),
            "AAPL": FakeHistory("AAPL", "stock:AAPL"),
        }),
        cognitive_system=SimpleNamespace(
            _price_history={"ETH": [1, 2], "MSFT": [3, 4]},
            _meta_domains={"crypto:SOL": {}, "stock:NVDA": {}},
        ),
    )

    await plugin2.restore_runtime_state(core2)
    derived = plugin2.export_runtime_state()
    assert set(derived["crypto_symbols"]) == {"BTC", "ETH", "SOL"}, derived
    assert set(derived["stock_symbols"]) == {"AAPL", "MSFT", "NVDA"}, derived

    await plugin2.persist_runtime_state(postgres2)
    saved_state = postgres2.saved[plugin2.STATE_CACHE_KEY]
    assert set(saved_state["crypto_symbols"]) == {"BTC", "ETH", "SOL"}, saved_state
    assert set(saved_state["stock_symbols"]) == {"AAPL", "MSFT", "NVDA"}, saved_state

    print("market_plugin_persistence_ok")


if __name__ == "__main__":
    asyncio.run(main())
