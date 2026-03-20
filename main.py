    async def _data_collection_loop(self):
        """
        Continuously collect market data and feed it into the cognitive system.
        Uses a SLIDING WINDOW fetcher to prevent overwhelming the cognitive core
        when tracking many symbols (e.g., 70+).
        """
        logger.info("Starting data collection loop (Sliding Window Mode)")
        
        # Max symbols to fetch per cycle to prevent "Cognitive Backlog"
        MAX_BATCH_SIZE = 20
        # Tracks our current position in the symbol rotation
        crypto_offset = 0
        stock_offset = 0

        while self.running:
            try:
                # 1. Get current full symbol list
                all_crypto = sorted(list(self.crypto_symbols))
                all_stocks = sorted(list(self.stock_symbols))
                
                if not all_crypto and not all_stocks:
                    logger.info("No symbols to fetch yet — waiting for market scan...")
                    await asyncio.sleep(10)
                    continue

                # 2. Select a subset (Sliding Window)
                # We take a mix of crypto and stocks, up to MAX_BATCH_SIZE
                crypto_count = min(len(all_crypto), MAX_BATCH_SIZE // 2)
                stock_count = min(len(all_stocks), MAX_BATCH_SIZE - crypto_count)
                
                # Adjust if one list is empty
                if crypto_count == 0: stock_count = min(len(all_stocks), MAX_BATCH_SIZE)
                if stock_count == 0: crypto_count = min(len(all_crypto), MAX_BATCH_SIZE)

                batch = []
                
                # Crypto slice
                if all_crypto:
                    for i in range(crypto_count):
                        idx = (crypto_offset + i) % len(all_crypto)
                        batch.append(all_crypto[idx])
                    crypto_offset = (crypto_offset + crypto_count) % len(all_crypto)

                # Stock slice
                if all_stocks:
                    for i in range(stock_count):
                        idx = (stock_offset + i) % len(all_stocks)
                        batch.append(all_stocks[idx])
                    stock_offset = (stock_offset + stock_count) % len(all_stocks)

                # 3. Fetch the batch
                logger.info(f"Fetching window [{len(batch)} symbols]: {batch[:10]}{'...' if len(batch) > 10 else ''}")
                ticks = await self.data_provider.fetch_batch(batch)

                # 4. Feed each tick into the cognitive system
                success_count = 0
                for tick in ticks:
                    if not tick or isinstance(tick, Exception):
                        continue

                    symbol = tick.get('symbol', '')
                    if self.data_provider.is_crypto(symbol):
                        domain = f"crypto:{symbol}"
                    else:
                        domain = f"stock:{symbol}"

                    # Ingest into the cognitive core (queues for cognitive loop)
                    await self.core.ingest(tick, domain)
                    success_count += 1

                    # Cache in Redis if available
                    if self.redis:
                        try:
                            await self.redis.cache_tick(symbol, tick)
                        except Exception:
                            pass

                if success_count > 0:
                    logger.info(f"Ingested {success_count}/{len(batch)} ticks (Window rotation: C:{crypto_offset} S:{stock_offset})")

                # 5. Wait for the next cycle
                # We use a shorter interval for the sliding window to keep data fresh
                # but overall throughput is much lower than fetching 70+ at once.
                interval = max(5, self.update_interval // 2) 
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(10)
