import functools
from datetime import datetime
from typing import Any, Optional
import pandas as pd
from cachetools import TTLCache
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CompanyCache:
    """
    Global in-memory cache for company DataFrames and LLM metadata.

    Key: sanitized_company_name (str)
    Rule: fetch ONCE per company per runtime; re-fetch only on TTL expiry or
          explicit invalidation.
    """

    def __init__(self, ttl_hours: int = 4):
        self._ttl_hours = ttl_hours
        self._store: TTLCache[str, tuple[pd.DataFrame, dict[str, Any], datetime]] = TTLCache(
            maxsize=10000,
            ttl=ttl_hours * 3600,
        )

    def get(self, company_name: str) -> Optional[tuple[pd.DataFrame, dict]]:
        """Return (df, metadata) if cached and not expired, else None."""
        entry = self._store.get(company_name)
        if entry is None:
            return None
        logger.debug("company_cache_hit", company=company_name)
        df, metadata, _ = entry
        return df, metadata

    def set(
        self,
        company_name: str,
        dataframe: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> None:
        """Store a company's DataFrame and metadata in the cache."""
        self._store[company_name] = (dataframe, metadata, datetime.utcnow())
        logger.info(
            "company_cache_set",
            company=company_name,
            rows=len(dataframe),
            ttl_hours=self._ttl_hours,
        )

    def info(self) -> dict:
        return {
            name: {
                "rows": len(entry[0]),
                "fetched_at": entry[2].isoformat(),
                "expired": False,
            }
            for name, entry in self._store.items()
        }


@functools.cache
def get_company_cache() -> CompanyCache:
    from src.config.settings import get_settings
    return CompanyCache(ttl_hours=get_settings().company_cache_ttl_hours)
