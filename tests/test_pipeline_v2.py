"""Tests for the refactored core pipeline system."""

import pytest
from unittest.mock import Mock, patch

from robodata.core.pipeline import (
    ProcessingConfig,
    ProcessingMode,
    ProcessingPipeline,
    Processor,
    BatchProcessor,
    CachingProcessor,
    AdaptiveProcessor,
    ProcessingResult,
)


class MockProcessor(Processor):
    """Mock processor for testing."""
    
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.setup_called = False
        self.cleanup_called = False
        self.processed_items = []
    
    def setup(self):
        self.setup_called = True
    
    def cleanup(self):
        self.cleanup_called = True
    
    def process_item(self, item):
        if self.should_fail:
            raise ValueError(f"Failed to process {item}")
        processed = f"processed_{item}"
        self.processed_items.append(processed)
        return processed


class MockDataSource:
    """Mock data source for testing."""
    
    def __init__(self, items):
        self.items = items
        self.iter_count = 0
    
    def __iter__(self):
        for item in self.items:
            self.iter_count += 1
            yield item
    
    def __len__(self):
        return len(self.items)


class MockDataSink:
    """Mock data sink for testing."""
    
    def __init__(self):
        self.written_items = []
        self.finalized = False
    
    def write(self, item, index):
        self.written_items.append((item, index))
    
    def finalize(self):
        self.finalized = True


class TestProcessingConfig:
    """Test ProcessingConfig class."""
    
    def test_default_config(self):
        config = ProcessingConfig()
        assert config.mode == ProcessingMode.SEQUENTIAL
        assert config.num_workers == 1
        assert config.chunk_size == 1000
        assert config.memory_limit_mb == 2048
        assert config.fail_fast is True
        assert config.progress_callback is None
    
    def test_custom_config(self):
        callback = Mock()
        config = ProcessingConfig(
            mode=ProcessingMode.PARALLEL,
            num_workers=4,
            chunk_size=500,
            memory_limit_mb=4096,
            fail_fast=False,
            progress_callback=callback
        )
        assert config.mode == ProcessingMode.PARALLEL
        assert config.num_workers == 4
        assert config.chunk_size == 500
        assert config.memory_limit_mb == 4096
        assert config.fail_fast is False
        assert config.progress_callback == callback


class TestProcessingPipeline:
    """Test ProcessingPipeline class."""
    
    def test_sequential_processing_success(self):
        processor = MockProcessor()
        source = MockDataSource(["item1", "item2", "item3"])
        sink = MockDataSink()
        config = ProcessingConfig(mode=ProcessingMode.SEQUENTIAL)
        
        pipeline = ProcessingPipeline(processor, config)
        result = pipeline.execute(source, sink)
        
        assert result.success is True
        assert result.items_processed == 3
        assert result.items_failed == 0
        assert len(result.errors) == 0
        assert processor.setup_called is True
        assert processor.cleanup_called is True
        assert sink.finalized is True
        assert len(sink.written_items) == 3
    
    def test_sequential_processing_with_failures(self):
        processor = MockProcessor(should_fail=True)
        source = MockDataSource(["item1", "item2"])
        sink = MockDataSink()
        config = ProcessingConfig(fail_fast=False)
        
        pipeline = ProcessingPipeline(processor, config)
        result = pipeline.execute(source, sink)
        
        assert result.success is False
        assert result.items_processed == 0
        assert result.items_failed == 2
        assert len(result.errors) == 2
        assert "Failed to process" in result.errors[0]
    
    def test_fail_fast_behavior(self):
        processor = MockProcessor(should_fail=True)
        source = MockDataSource(["item1", "item2", "item3"])
        sink = MockDataSink()
        config = ProcessingConfig(fail_fast=True)
        
        pipeline = ProcessingPipeline(processor, config)
        result = pipeline.execute(source, sink)
        
        assert result.success is False
        assert result.items_failed == 1  # Should stop after first failure
    
    def test_progress_callback(self):
        callback = Mock()
        processor = MockProcessor()
        source = MockDataSource(["item1", "item2"])
        sink = MockDataSink()
        config = ProcessingConfig(progress_callback=callback)
        
        pipeline = ProcessingPipeline(processor, config)
        pipeline.execute(source, sink)
        
        # Check that progress callback was called
        assert callback.call_count >= 2  # Called for each item
    
    def test_no_sink_processing(self):
        processor = MockProcessor()
        source = MockDataSource(["item1", "item2"])
        config = ProcessingConfig()
        
        pipeline = ProcessingPipeline(processor, config)
        result = pipeline.execute(source, sink=None)
        
        assert result.success is True
        assert result.items_processed == 2
        assert processor.setup_called is True
        assert processor.cleanup_called is True


class TestBatchProcessor:
    """Test BatchProcessor class."""
    
    def test_batch_processing(self):
        class TestBatchProcessor(BatchProcessor):
            def __init__(self, batch_size=2):
                super().__init__(batch_size)
                self.batches_processed = []
            
            def _process_batch(self, batch):
                processed_batch = [f"batch_{item}" for item in batch]
                self.batches_processed.append(processed_batch)
                return processed_batch
        
        processor = TestBatchProcessor(batch_size=2)
        processor.setup()
        
        # Process items
        result1 = processor.process_item("item1")
        result2 = processor.process_item("item2")
        result3 = processor.process_item("item3")
        
        # First batch should be ready after 2 items
        assert result1 is None
        assert result2 is not None
        assert result2 == ["batch_item1", "batch_item2"]
        
        # Third item should not trigger batch yet
        assert result3 is None
        
        # Finalize remaining batch
        final_batch = processor.finalize_batch()
        assert final_batch == ["batch_item3"]
        
        assert len(processor.batches_processed) == 2
        processor.cleanup()


class TestCachingProcessor:
    """Test CachingProcessor class."""
    
    def test_cache_functionality(self):
        class TestCachingProcessor(CachingProcessor):
            def __init__(self, cache_size=2):
                super().__init__(cache_size)
                self.process_count = {}
            
            def _process_uncached(self, item):
                self.process_count[item] = self.process_count.get(item, 0) + 1
                return f"cached_{item}"
            
            def _get_cache_key(self, item):
                return str(item)
        
        processor = TestCachingProcessor(cache_size=2)
        processor.setup()
        
        # Process same item multiple times
        result1 = processor.process_item("item1")
        result2 = processor.process_item("item1")  # Should use cache
        result3 = processor.process_item("item2")
        result4 = processor.process_item("item1")  # Should still use cache
        
        assert result1 == "cached_item1"
        assert result2 == "cached_item1"
        assert result3 == "cached_item2"
        assert result4 == "cached_item1"
        
        # item1 should only be processed once due to caching
        assert processor.process_count["item1"] == 1
        assert processor.process_count["item2"] == 1
        
        processor.cleanup()
    
    def test_cache_eviction(self):
        class TestCachingProcessor(CachingProcessor):
            def _process_uncached(self, item):
                return f"cached_{item}"
            
            def _get_cache_key(self, item):
                return str(item)
        
        processor = TestCachingProcessor(cache_size=2)
        processor.setup()
        
        # Fill cache
        processor.process_item("item1")
        processor.process_item("item2")
        processor.process_item("item3")  # Should evict item1
        
        # Process item1 again - should be reprocessed
        result = processor.process_item("item1")
        assert result == "cached_item1"
        
        processor.cleanup()


class TestAdaptiveProcessor:
    """Test AdaptiveProcessor class."""
    
    def test_adaptive_behavior(self):
        class TestAdaptiveProcessor(AdaptiveProcessor):
            def __init__(self, base_processor):
                super().__init__(base_processor)
                self.adaptation_applied_to = None
            
            def _adapt_to_item(self, item):
                self.adaptation_applied_to = item
        
        base_processor = MockProcessor()
        adaptive_processor = TestAdaptiveProcessor(base_processor)
        
        adaptive_processor.setup()
        
        # Process first item - should trigger adaptation
        result1 = adaptive_processor.process_item("item1")
        assert adaptive_processor.adaptation_applied_to == "item1"
        assert result1 == "processed_item1"
        
        # Process second item - should not trigger adaptation again
        result2 = adaptive_processor.process_item("item2")
        assert adaptive_processor.adaptation_applied_to == "item1"  # Still first item
        assert result2 == "processed_item2"
        
        adaptive_processor.cleanup()


class TestPipelineIntegration:
    """Test pipeline integration scenarios."""
    
    def test_nested_processors(self):
        """Test using processors in combination."""
        # Create a chain of processors
        base_processor = MockProcessor()
        caching_processor = CachingProcessor(cache_size=10)
        adaptive_processor = AdaptiveProcessor(base_processor)
        
        # This would need a composition strategy in real implementation
        # For now, test that they can be created and configured
        assert caching_processor.cache_size == 10
        assert adaptive_processor.base_processor == base_processor
    
    def test_error_propagation(self):
        """Test error handling in pipeline."""
        processor = MockProcessor(should_fail=True)
        source = MockDataSource(["item1"])
        config = ProcessingConfig(fail_fast=True)
        
        pipeline = ProcessingPipeline(processor, config)
        
        with pytest.raises(ValueError, match="Failed to process"):
            pipeline.execute(source)
    
    def test_large_dataset_processing(self):
        """Test processing of larger datasets."""
        processor = MockProcessor()
        source = MockDataSource([f"item_{i}" for i in range(1000)])
        config = ProcessingConfig()
        
        pipeline = ProcessingPipeline(processor, config)
        result = pipeline.execute(source)
        
        assert result.success is True
        assert result.items_processed == 1000
        assert result.items_failed == 0


if __name__ == "__main__":
    pytest.main([__file__])
