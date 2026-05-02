# CERN-First Mesh Pivot

The mesh now boots with real CERN CMS collision data instead of financial market data.  This is not a cosmetic rename; it changes the default proving domain while preserving the market stack as an optional legacy plugin.

| Area | Change |
|---|---|
| Default source | `CERNCollisionPlugin` loads CERN Open Data record 304 by default. |
| Market ingestion | `MarketPlugin` loads only with `ENABLE_MARKET_PLUGIN=1`. |
| Market-context plugins | Sentiment, macro, on-chain, news, derivatives, social, and microstructure plugins load only with `ENABLE_MARKET_CONTEXT_PLUGINS=1`. |
| Observation contract | The plugin emits `(observation, domain)` tuples using `entity_id`, `value`, `secondary_value`, and metadata. |
| CERN domain | `cern:cms:dielectron`. |

The default CERN dataset is `Events with two electrons from 2010`, DOI `10.7483/OPENDATA.CMS.PCSW.AHVG`, downloaded from `https://opendata.cern.ch/record/304/files/dielectron.csv?download=1`.

## Runtime knobs

```env
DISABLE_CERN_PLUGIN=false
CERN_COLLISION_DATA_URL=https://opendata.cern.ch/record/304/files/dielectron.csv?download=1
CERN_COLLISION_BATCH_SIZE=25
CERN_COLLISION_PRIMARY_VALUE=M
CERN_COLLISION_SECONDARY_VALUE=pt1
ENABLE_MARKET_PLUGIN=false
ENABLE_MARKET_CONTEXT_PLUGINS=false
```

## Next deeper cleanup

Several historical names remain in deeper layers for backward compatibility, especially `price_history`, `symbol`, market dashboards, and some persistence table names.  The default runtime no longer depends on those market feeds, but a future schema migration should rename them to `observation_history`, `entity_id`, and generic observation tables.
