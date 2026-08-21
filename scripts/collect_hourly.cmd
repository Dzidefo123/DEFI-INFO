@echo off
REM Scheduled on-chain collection for DEFI-INFO.
REM Registered as the Windows task "DEFI-INFO Collect". Read-only: it fetches
REM current market readings and appends them to .features.sqlite. Writes are
REM idempotent, so an overlapping or retried run stores nothing twice.
REM
REM Cadence matters. The risk engine needs 8 readings per series before it will
REM score anything, and a baseline only means something if its readings are
REM evenly spaced. Changing the interval changes the unit every anomaly is
REM implicitly measured in. See src/blockchain/collect.py.
REM
REM hyperliquid (market metrics) and hyperevm (JSON-RPC chain reads) are
REM collected. Ethena still has no on-chain reader and says so rather than
REM being skipped. A hyperevm run scans back from the chain head until it has
REM enough of both block types, so it costs roughly 60 requests against a
REM 100-per-minute public limit -- do not shorten the interval below hourly
REM without checking that budget.

setlocal
set "PROJECT=%~dp0.."
REM Interpreter: set DEFI_PYTHON to a full path if the scheduled task runs
REM without the PATH you expect. Otherwise the one on PATH is used.
if defined DEFI_PYTHON (set "PY=%DEFI_PYTHON%") else (set "PY=python")
set "LOG=%PROJECT%\.collect.log"

cd /d "%PROJECT%" || exit /b 1

>>"%LOG%" echo.
>>"%LOG%" echo === %DATE% %TIME% ===
REM Arguments pass through: run this with --dry-run to exercise the whole
REM path, network calls included, without writing to the feature store.
"%PY%" -m src.blockchain.collect --protocol hyperliquid --protocol hyperevm %* >>"%LOG%" 2>&1
>>"%LOG%" echo exit=%ERRORLEVEL%
exit /b %ERRORLEVEL%
