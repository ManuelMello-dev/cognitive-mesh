# Persistence Diagnosis for `cognitive-mesh`

I reviewed the repository, the attached runtime context, and the current hosting behavior. Your complaint is correct: **if this system is meant to be continuously learning, then a cold wake-up model is not acceptable**. A service that sleeps and then reconstructs itself only when you open it is not behaving like a true always-on cognitive process.

## What is actually going wrong

There are **two separate failure modes**, and they reinforce each other.

| Layer | Problem | Practical effect |
|---|---|---|
| Hosting layer | Railway can sleep a service when serverless mode is enabled and the service is inactive by Railway's outbound-traffic rules. The next request wakes it and incurs a cold boot.[1] | The process is not truly alive 24/7. In-memory state disappears whenever the container is stopped. |
| Application layer | The repository had **configuration mismatches** that could silently disable persistence even when a database or cache existed. | After a restart, the mesh can come back looking fresh because state restoration was never actually wired up. |

The hosting issue explains why the process can appear dead between sessions. The application issue explains why it can also come back **empty** instead of resuming from prior memory.

## The concrete repository bug I found

The codebase was inconsistent about persistence environment variables.

| Component | Expected by code before fix | Documented or commonly provided variable | Why this breaks persistence |
|---|---|---|---|
| PostgreSQL | `POSTGRES_URL` | `.env.example` used `DATABASE_URL`, and managed platforms commonly inject `DATABASE_URL` | The startup path in `main.py` only initialized Postgres when `Config.POSTGRES_URL` was present, so a valid `DATABASE_URL` alone could leave Postgres disabled. |
| Redis | `REDIS_URL` in config | `.env.example` documented `REDIS_HOST` and `REDIS_PORT` | The runtime only looked for `REDIS_URL`, but the Redis client constructor was written more like a host-based connector. This mismatch could silently skip Redis or build the wrong connection string. |

The most important one is the PostgreSQL mismatch. The persistence code already knows how to save and load long-term state, but it only runs when Postgres is connected. Because the repository mixed `POSTGRES_URL` and `DATABASE_URL`, the deployment could look correctly configured while actually never restoring state.

## Evidence from the code

The startup path in `main.py` initializes databases and then calls `load_state()`. That means the intended architecture is already based on restore-on-start behavior, not a stateless rebuild. However, that restore path only becomes active when the persistence backend is successfully connected.

In `.env.example`, PostgreSQL is described as **required for persistent memory**, which matches the actual design. In `core/distributed_core.py`, the save and restore logic is guarded behind `if self.postgres:` checks, confirming that without a working Postgres connection, learned state is effectively process-local RAM.

## External platform evidence

Railway's current documentation states that when serverless mode is enabled, a service is considered inactive if it sends no outbound packets for more than 10 minutes, and the next request wakes it up with a cold boot.[1] Railway also states that app sleeping is disabled by default, but it exists as a supported feature.[2]

> "If no packets are sent from the service for over 10 minutes, the service is considered inactive." — Railway Docs, *Serverless*[1]

> "The first request made to a slept service wakes it. It may take a small amount of time for the service to spin up again on the first request." — Railway Docs, *Serverless*[1]

So the platform behavior and the code behavior are distinct. Even a correct application can still get put to sleep if the deployment mode allows it. And even a continuously running process can still lose its learned state if persistence is not actually connected.

## Fixes I applied in the repository

I implemented the configuration fixes directly in the checked-out repository so the application is less likely to fall back to fake persistence.

| File | Change |
|---|---|
| `config/config.py` | Added fallback from `POSTGRES_URL` to `DATABASE_URL`, and added Redis support for both URL and host/port configuration. |
| `main.py` | Updated Redis initialization so it accepts either `REDIS_URL` or `REDIS_HOST`/`REDIS_PORT`. |
| `storage/redis_cache.py` | Updated the Redis connector so it can parse a full `redis://...` URL or build one from host and port. |
| `.env.example` | Corrected the documented persistence variables and clarified which forms are accepted. |
| `README.md` | Updated the persistence section to reflect the fixed runtime behavior. |

I also ran a syntax validation with `python3.11 -m py_compile` against the modified Python files, and that validation passed.

## What this means operationally

A true 24/7 system requires **both** of the following conditions:

| Requirement | Why it matters |
|---|---|
| The host must keep the service alive continuously | Otherwise RAM disappears whenever the container is slept, rebuilt, or restarted. |
| The application must persist cognitive state externally | Otherwise any restart, even an unavoidable one, wipes learning and forces a fresh start. |

That means the correct architecture is **not** cron, and your objection was justified. Cron only schedules periodic execution. It does not provide a continuously alive process, and it does not preserve in-memory cognition between invocations.

## The correct target architecture

If the goal is "always learning, always on," then the service should be treated as a resident process with durable external state.

| Concern | Correct approach |
|---|---|
| Continuous runtime | Deploy as a non-sleeping long-lived service rather than a scheduled job. |
| Memory continuity | Ensure PostgreSQL is actually connected so `load_state()` and checkpointing work across restarts. |
| Cache continuity | Optionally connect Redis for low-latency state sharing and caching. |
| Crash recovery | Keep checkpointing enabled so unavoidable restarts resume from durable state instead of starting blank. |

## What you should change in the deployment

First, confirm that the deployed service is **not** running with serverless sleep semantics. If it is allowed to sleep, it will never satisfy the requirement that it remain alive at all times.[1]

Second, ensure the deployment provides one of these for PostgreSQL:

| Accepted variable after fix | Notes |
|---|---|
| `POSTGRES_URL` | Explicit repository-native variable name |
| `DATABASE_URL` | Common managed-provider default; now accepted by the code |

Third, verify that the running service logs actually show a successful PostgreSQL connection during startup. If the logs do not report a successful connection, then persistence is still not truly active and a restart will still feel like a reset.

## Bottom line

Your diagnosis was directionally right. The problem is **not** that the system merely needs a timer. The real issue is that it was behaving like a wake-on-demand service while also carrying a persistence configuration bug that could prevent it from restoring its learned state. In other words, it was broken in exactly the way you described: not continuously alive, and prone to coming back fresh.

I have fixed the repository-side configuration mismatch. The remaining requirement is deployment-side: the service must be run in a truly non-sleeping mode, and PostgreSQL must be present in the environment so the restored state path is actually exercised.

## References

[1]: https://docs.railway.com/deployments/serverless "Railway Docs — Serverless"
[2]: https://station.railway.com/questions/sleep-app-on-inactivity-2cd0b079 "Railway Central Station — sleep app on inactivity"
