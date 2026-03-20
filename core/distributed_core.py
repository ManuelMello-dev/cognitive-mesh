    def _cognitive_loop(self):
        """
        Main cognitive loop running in a dedicated thread.
        Optimized with dynamic batching to handle bursts of data without 
        causing 502/499 timeouts in the HTTP server.
        """
        logger.info("Cognitive loop thread started (Dynamic Batching Mode)")
        iteration = 0
        last_cache_update = 0
        
        while self._running:
            try:
                iteration += 1
                start_time = time.time()
                
                # 1. Process pending observations with DYNAMIC BATCHING
                # If the queue is backed up, we process more items per loop
                queue_size = len(self._pending_observations)
                batch_size = 10
                if queue_size > 50: batch_size = 20
                if queue_size > 200: batch_size = 50
                
                batch = []
                while self._pending_observations and len(batch) < batch_size:
                    batch.append(self._pending_observations.popleft())

                if batch:
                    for obs, domain in batch:
                        try:
                            # Process through cognitive engines
                            self.cognitive_system.process_observation(obs, domain)
                            
                            # Feed into prediction engine
                            symbol = obs.get('symbol')
                            price = obs.get('price')
                            if symbol and price:
                                self.prediction_engine.record_observation(obs, domain)
                        except Exception as e:
                            self._errors += 1
                            logger.error(f"Error processing observation: {e}")

                # 1b. Update state cache every 5s OR if queue is large
                # Only update if we've actually processed something new
                now = time.time()
                if (now - last_cache_update > 5.0) or (queue_size > 100 and iteration % 20 == 0):
                    self._update_state_cache()
                    last_cache_update = now

                # 2. Perform periodic maintenance (every ~10s)
                if iteration % 100 == 0:
                    with self._lock:
                        # Feed prediction accuracy back to rules
                        self._feed_rule_confidence_back()
                        
                        # Converge concepts (merge/prune)
                        self._run_concept_convergence()

                # 3. Deep Introspection (every ~30s)
                if iteration % 300 == 0:
                    with self._lock:
                        try:
                            # Trigger goal formation
                            ctx = self._build_goal_context()
                            self.cognitive_system.goals.generate_goals(ctx)
                        except Exception as e:
                            logger.error(f"Error in introspection: {e}")

                # Performance monitoring
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    logger.warning(f"Cognitive loop slow: {elapsed:.2f}s (Batch: {len(batch)}, Queue: {queue_size})")

                # Adaptive sleep: sleep less if the queue is backed up
                sleep_time = 0.1
                if queue_size > 100: sleep_time = 0.01
                if queue_size > 500: sleep_time = 0.001
                
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Critical error in cognitive loop: {e}")
                time.sleep(1)
