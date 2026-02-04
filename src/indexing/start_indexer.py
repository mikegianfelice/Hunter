#!/usr/bin/env python3
"""
Start the swap indexer as a background service.

This can be run standalone or integrated into the main trading loop.
"""

import logging
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.indexing.swap_indexer import get_indexer
from src.config.config_loader import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Start the swap indexer"""
    # Check if indexing is enabled
    enabled = get_config("swap_indexer.enabled", False)
    if not enabled:
        logger.info("Swap indexer is disabled in config")
        return
    
    indexer = get_indexer()
    
    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        logger.info("Shutting down swap indexer...")
        indexer.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start indexing
    indexer.start()
    
    # Keep running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
