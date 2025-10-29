#!/usr/bin/env python3
"""
Advanced Caching System - Phase 3
High-performance caching with TTL, LRU eviction, and memory management
"""

import time
import json
import pickle
import hashlib
import asyncio
import aiofiles
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, Callable, List
from dataclasses import dataclass, field
from collections import OrderedDict
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    max_memory_items: int = 1000
    max_disk_items: int = 10000
    default_ttl: int = 300  # 5 minutes
    cleanup_interval: int = 60  # 1 minute
    enable_disk_cache: bool = True
    cache_directory: str = "cache"
    compression_enabled: bool = True
    enable_metrics: bool = True

@dataclass
class CacheItem:
    """Individual cache item with metadata"""
    key: str
    value: Any
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    compressed: bool = False

class CacheMetrics:
    """Cache performance metrics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.disk_reads = 0
        self.disk_writes = 0
        self.compression_saves = 0
        self.errors = 0
        self.total_requests = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 1.0 - self.hit_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "disk_reads": self.disk_reads,
            "disk_writes": self.disk_writes,
            "compression_saves": self.compression_saves,
            "errors": self.errors,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate
        }

class AdvancedCache:
    """
    Advanced caching system with:
    - Memory and disk caching
    - TTL and LRU eviction
    - Compression and serialization
    - Async support
    - Performance metrics
    - Automatic cleanup
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.memory_cache: OrderedDict[str, CacheItem] = OrderedDict()
        self.disk_cache_dir = Path(self.config.cache_directory)
        self.metrics = CacheMetrics()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Create cache directory if needed
        if self.config.enable_disk_cache:
            self.disk_cache_dir.mkdir(exist_ok=True)
    
    async def start(self):
        """Start the cache system and cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Advanced cache system started")
    
    async def stop(self):
        """Stop the cache system and cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Advanced cache system stopped")
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of a value in bytes"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float, bool)):
                return 8
            else:
                # Serialize to get size
                serialized = pickle.dumps(value)
                return len(serialized)
        except Exception:
            return 1024  # Default estimate
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            if isinstance(value, (str, int, float, bool, list, dict)):
                return pickle.dumps(value)
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return pickle.dumps(str(value))
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
    
    def _get_disk_path(self, key: str) -> Path:
        """Get disk cache path for a key"""
        # Create hash for filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.disk_cache_dir / f"{key_hash}.cache"
    
    async def _save_to_disk(self, item: CacheItem) -> bool:
        """Save item to disk cache"""
        if not self.config.enable_disk_cache:
            return False
        
        try:
            disk_path = self._get_disk_path(item.key)
            
            # Prepare data for disk storage
            disk_data = {
                "key": item.key,
                "value": item.value,
                "created_at": item.created_at,
                "expires_at": item.expires_at,
                "access_count": item.access_count,
                "last_accessed": item.last_accessed,
                "compressed": item.compressed
            }
            
            # Serialize and save
            serialized_data = self._serialize_value(disk_data)
            
            async with aiofiles.open(disk_path, 'wb') as f:
                await f.write(serialized_data)
            
            self.metrics.disk_writes += 1
            return True
            
        except Exception as e:
            logger.error(f"Error saving to disk: {e}")
            self.metrics.errors += 1
            return False
    
    async def _load_from_disk(self, key: str) -> Optional[CacheItem]:
        """Load item from disk cache"""
        if not self.config.enable_disk_cache:
            return None
        
        try:
            disk_path = self._get_disk_path(key)
            
            if not disk_path.exists():
                return None
            
            async with aiofiles.open(disk_path, 'rb') as f:
                serialized_data = await f.read()
            
            disk_data = self._deserialize_value(serialized_data)
            if not disk_data:
                return None
            
            # Recreate CacheItem
            item = CacheItem(
                key=disk_data["key"],
                value=disk_data["value"],
                created_at=disk_data["created_at"],
                expires_at=disk_data["expires_at"],
                access_count=disk_data["access_count"],
                last_accessed=disk_data["last_accessed"],
                compressed=disk_data.get("compressed", False)
            )
            
            item.size_bytes = self._calculate_size(item.value)
            
            self.metrics.disk_reads += 1
            return item
            
        except Exception as e:
            logger.error(f"Error loading from disk: {e}")
            self.metrics.errors += 1
            return None
    
    async def _evict_lru(self) -> int:
        """Evict least recently used items from memory cache"""
        evicted = 0
        
        while len(self.memory_cache) > self.config.max_memory_items:
            if not self.memory_cache:
                break
            
            # Remove least recently used item
            key, item = self.memory_cache.popitem(last=False)
            
            # Try to save to disk before evicting
            if self.config.enable_disk_cache:
                await self._save_to_disk(item)
            
            evicted += 1
            self.metrics.evictions += 1
        
        return evicted
    
    async def _cleanup_expired(self) -> int:
        """Remove expired items from memory cache"""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self.memory_cache.items():
            if current_time > item.expires_at:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        return len(expired_keys)
    
    async def _cleanup_loop(self):
        """Background cleanup task"""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                async with self._lock:
                    # Cleanup expired items
                    expired = await self._cleanup_expired()
                    
                    # Evict LRU items if needed
                    evicted = await self._evict_lru()
                    
                    if expired > 0 or evicted > 0:
                        logger.debug(f"Cache cleanup: {expired} expired, {evicted} evicted")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                self.metrics.errors += 1
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self.metrics.total_requests += 1
        
        async with self._lock:
            # Check memory cache first
            if key in self.memory_cache:
                item = self.memory_cache[key]
                
                # Check if expired
                if time.time() > item.expires_at:
                    del self.memory_cache[key]
                    self.metrics.misses += 1
                    return None
                
                # Update access info
                item.access_count += 1
                item.last_accessed = time.time()
                
                # Move to end (most recently used)
                self.memory_cache.move_to_end(key)
                
                self.metrics.hits += 1
                return item.value
            
            # Try disk cache
            item = await self._load_from_disk(key)
            if item:
                # Check if expired
                if time.time() > item.expires_at:
                    # Remove from disk
                    disk_path = self._get_disk_path(key)
                    if disk_path.exists():
                        disk_path.unlink()
                    self.metrics.misses += 1
                    return None
                
                # Load into memory cache
                self.memory_cache[key] = item
                self.memory_cache.move_to_end(key)
                
                self.metrics.hits += 1
                return item.value
            
            self.metrics.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if ttl is None:
            ttl = self.config.default_ttl
        
        current_time = time.time()
        expires_at = current_time + ttl
        
        # Calculate size
        size_bytes = self._calculate_size(value)
        
        # Create cache item
        item = CacheItem(
            key=key,
            value=value,
            created_at=current_time,
            expires_at=expires_at,
            size_bytes=size_bytes
        )
        
        async with self._lock:
            # Store in memory cache
            self.memory_cache[key] = item
            self.memory_cache.move_to_end(key)
            
            # Evict if needed
            await self._evict_lru()
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        async with self._lock:
            # Remove from memory
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            # Remove from disk
            if self.config.enable_disk_cache:
                disk_path = self._get_disk_path(key)
                if disk_path.exists():
                    disk_path.unlink()
            
            return True
    
    async def clear(self) -> int:
        """Clear all cache data"""
        async with self._lock:
            # Clear memory cache
            memory_count = len(self.memory_cache)
            self.memory_cache.clear()
            
            # Clear disk cache
            disk_count = 0
            if self.config.enable_disk_cache and self.disk_cache_dir.exists():
                for cache_file in self.disk_cache_dir.glob("*.cache"):
                    cache_file.unlink()
                    disk_count += 1
            
            return memory_count + disk_count
    
    async def get_or_set(self, key: str, factory: Callable[[], Any], ttl: Optional[int] = None) -> Any:
        """Get value from cache or set it using factory function"""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Generate value using factory
        try:
            if asyncio.iscoroutinefunction(factory):
                value = await factory()
            else:
                value = factory()
            
            await self.set(key, value, ttl)
            return value
            
        except Exception as e:
            logger.error(f"Factory function error for key {key}: {e}")
            self.metrics.errors += 1
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "memory_items": len(self.memory_cache),
            "max_memory_items": self.config.max_memory_items,
            "disk_enabled": self.config.enable_disk_cache,
            "metrics": self.metrics.to_dict()
        }
    
    async def warm_up(self, warm_up_data: Dict[str, Any]):
        """Warm up cache with initial data"""
        logger.info(f"Warming up cache with {len(warm_up_data)} items")
        
        for key, value in warm_up_data.items():
            await self.set(key, value)
        
        logger.info("Cache warm-up complete")

# Global cache instance
_global_cache: Optional[AdvancedCache] = None

async def get_cache() -> AdvancedCache:
    """Get global cache instance"""
    global _global_cache
    if _global_cache is None:
        config = CacheConfig(
            max_memory_items=1000,
            max_disk_items=10000,
            default_ttl=300,
            enable_disk_cache=True,
            cache_directory="cache"
        )
        _global_cache = AdvancedCache(config)
        await _global_cache.start()
    return _global_cache

async def cache_get(key: str) -> Optional[Any]:
    """Get value from global cache"""
    cache = await get_cache()
    return await cache.get(key)

async def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Set value in global cache"""
    cache = await get_cache()
    return await cache.set(key, value, ttl)

async def cache_get_or_set(key: str, factory: Callable[[], Any], ttl: Optional[int] = None) -> Any:
    """Get value from cache or set it using factory function"""
    cache = await get_cache()
    return await cache.get_or_set(key, factory, ttl)

async def cache_clear():
    """Clear global cache"""
    cache = await get_cache()
    return await cache.clear()

async def cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    cache = await get_cache()
    return cache.get_stats()
