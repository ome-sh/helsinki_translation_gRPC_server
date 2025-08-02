# License
#
# This software is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
#
# Copyright (C) 2025 Roland Kohlhuber
#
# **Note:** The AI models used by this software (Helsinki-NLP/opus-mt-es-en, Helsinki-NLP/opus-tatoeba-es-zh) retain their original licenses and are not subject to the AGPL license terms.
#
# For the complete license text, see: https://www.gnu.org/licenses/agpl-3.0.html

import grpc
import logging
import time
from concurrent import futures
from typing import List, Tuple, Any
from dataclasses import dataclass
from threading import Lock, Thread
import translation_pb2
import translation_pb2_grpc
from transformers import MarianMTModel, MarianTokenizer
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BatchItem:
    """Represents a single translation request in a batch."""
    text: str
    future: futures.Future
    timestamp: float
    target_language: str

class DynamicBatcher:
    """
    Dynamic batching manager that collects requests and processes them in batches
    with a maximum wait time of 50ms.
    """
    
    def __init__(self, max_batch_size: int = 128, max_wait_time: float = 0.05):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time  # 50ms
        self.pending_requests = []
        self.lock = Lock()
        self.translation_service = None  # Will be set by TranslationService
        # Use a persistent thread pool for processing batches
        self.executor = futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix="BatchProcessor")
        
    def add_request(self, text: str, target_language: str) -> futures.Future:
        """Add a translation request to the batch queue."""
        future = futures.Future()
        item = BatchItem(
            text=text,
            future=future,
            timestamp=time.time(),
            target_language=target_language
        )
        
        batch_to_process = None
        with self.lock:
            self.pending_requests.append(item)
            
            should_process = (
                len(self.pending_requests) >= self.max_batch_size or
                (self.pending_requests and
                 time.time() - self.pending_requests[0].timestamp >= self.max_wait_time)
            )
            
            if should_process:
                batch_to_process = self.pending_requests[:]
                self.pending_requests.clear()

        if batch_to_process:
            self.executor.submit(self._process_batch, batch_to_process)
            
        return future
    
    def _maybe_process_batch_by_time(self):
        """
        Periodically checks if the batch should be processed due to timeout.
        This runs in a background thread.
        """
        batch_to_process = None
        with self.lock:
            if not self.pending_requests:
                return
            
            # Process only if the timeout is met and the batch isn't empty
            if time.time() - self.pending_requests[0].timestamp >= self.max_wait_time:
                batch_to_process = self.pending_requests[:]
                self.pending_requests.clear()
        
        if batch_to_process:
            self.executor.submit(self._process_batch, batch_to_process)
            
    def _process_batch(self, batch_items: List[BatchItem]):
        """Process a batch of translation requests."""
        try:
            # Group by target language
            en_items = [item for item in batch_items if item.target_language == 'en']
            zh_items = [item for item in batch_items if item.target_language == 'zh']
            
            # Process each language group
            if en_items:
                self.translation_service._process_language_batch(en_items, 'en')
            if zh_items:
                self.translation_service._process_language_batch(zh_items, 'zh')
                
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            # Set error for all items in batch
            for item in batch_items:
                if not item.future.done():
                    item.future.set_exception(e)

class TranslationService(translation_pb2_grpc.TranslationServiceServicer):
    """
    gRPC service for translating Spanish text to English and Chinese
    using pre-trained MarianMT models with dynamic batching support.
    """
    
    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.05):
        """Initialize the translation models and batching system."""
        logger.info("Initializing translation service with dynamic batching...")
        
        try:
            # Device detection and configuration
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load Spanish to English model and tokenizer
            self.es_en_model_name = "Helsinki-NLP/opus-mt-es-en"
            logger.info(f"Loading Spanish to English model: {self.es_en_model_name}")
            self.es_en_tokenizer = MarianTokenizer.from_pretrained(self.es_en_model_name)
            self.es_en_model = MarianMTModel.from_pretrained(self.es_en_model_name)
            
            # Load Spanish to Chinese model and tokenizer
            self.es_zh_model_name = "Helsinki-NLP/opus-tatoeba-es-zh"
            logger.info(f"Loading Spanish to Chinese model: {self.es_zh_model_name}")
            self.es_zh_tokenizer = MarianTokenizer.from_pretrained(self.es_zh_model_name)
            self.es_zh_model = MarianMTModel.from_pretrained(self.es_zh_model_name)
            
            # Move models to the appropriate device
            self.es_en_model = self.es_en_model.to(self.device)
            self.es_zh_model = self.es_zh_model.to(self.device)
            
            # Set models to evaluation mode
            self.es_en_model.eval()
            self.es_zh_model.eval()
            
            # Initialize dynamic batcher
            self.batcher = DynamicBatcher(max_batch_size, max_wait_time)
            # Set reference to this service for batch processing
            self.batcher.translation_service = self
            
            # Start the batch processing timer
            self._start_batch_timer()
            
            logger.info(f"Translation service initialized successfully!")
            logger.info(f"Batching config: max_batch_size={max_batch_size}, max_wait_time={max_wait_time*1000}ms")
            
        except Exception as e:
            logger.error(f"Failed to initialize translation service: {str(e)}")
            raise
    
    def _start_batch_timer(self):
        """Start a timer that periodically checks for batch processing."""
        def timer_loop():
            while True:
                time.sleep(0.01)  # Check every 10ms
                self.batcher._maybe_process_batch_by_time()
        
        # Start timer in daemon thread
        timer_thread = Thread(target=timer_loop, daemon=True)
        timer_thread.start()
    
    def _process_language_batch(self, items: List[BatchItem], target_language: str):
        """Process a batch of translation requests for a specific language."""
        try:
            texts = [item.text for item in items]
            logger.info(f"Processing batch of {len(texts)} items for {target_language}")
            
            # Select appropriate model and tokenizer
            if target_language == 'en':
                tokenizer = self.es_en_tokenizer
                model = self.es_en_model
            else:  # zh
                tokenizer = self.es_zh_tokenizer
                model = self.es_zh_model
            
            # Batch tokenization
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move input tensors to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Batch translation
            with torch.no_grad():
                translated = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    length_penalty=0.6,
                    early_stopping=True
                )
            
            # Decode translations
            translated_texts = []
            for i in range(translated.shape[0]):
                translated_text = tokenizer.decode(
                    translated[i],
                    skip_special_tokens=True
                )
                translated_texts.append(translated_text)
            
            # Set results for all futures
            for item, translated_text in zip(items, translated_texts):
                if not item.future.done():
                    item.future.set_result(translated_text)
            
            logger.info(f"Batch processing completed for {len(items)} items")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            # Set error for all items in batch
            for item in items:
                if not item.future.done():
                    item.future.set_exception(e)
    
    def TranslateToEnglish(self, request, context):
        """
        Translate Spanish text to English using dynamic batching.
        
        Args:
            request: gRPC request containing the text to translate
            context: gRPC context
            
        Returns:
            TranslationResponse containing the translated text
        """
        try:
            text = request.text.strip()
            
            if not text:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Input text cannot be empty")
                return translation_pb2.TranslationResponse()
            
            logger.debug(f"Queuing English translation: '{text[:50]}...'")
            
            # Add to batch queue
            future = self.batcher.add_request(text, 'en')
            
            # Wait for result (with timeout)
            try:
                # Changed timeout to a more reasonable value
                translated_text = future.result(timeout=10.0) 
                
                logger.debug(f"Translation completed: '{translated_text[:50]}...'")
                return translation_pb2.TranslationResponse(translated_text=translated_text)
                
            except futures.TimeoutError:
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                context.set_details("Translation timeout")
                return translation_pb2.TranslationResponse()
            
        except Exception as e:
            logger.error(f"Error in TranslateToEnglish: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Translation failed: {str(e)}")
            return translation_pb2.TranslationResponse()
    
    def TranslateToChinese(self, request, context):
        """
        Translate Spanish text to Chinese using dynamic batching.
        
        Args:
            request: gRPC request containing the text to translate
            context: gRPC context
            
        Returns:
            TranslationResponse containing the translated text
        """
        try:
            text = request.text.strip()
            
            if not text:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Input text cannot be empty")
                return translation_pb2.TranslationResponse()
            
            logger.debug(f"Queuing Chinese translation: '{text[:50]}...'")
            
            # Add to batch queue
            future = self.batcher.add_request(text, 'zh')
            
            # Wait for result (with timeout)
            try:
                # Changed timeout to a more reasonable value
                translated_text = future.result(timeout=10.0)
                
                logger.debug(f"Translation completed: '{translated_text[:50]}...'")
                return translation_pb2.TranslationResponse(translated_text=translated_text)
                
            except futures.TimeoutError:
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                context.set_details("Translation timeout")
                return translation_pb2.TranslationResponse()
            
        except Exception as e:
            logger.error(f"Error in TranslateToChinese: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Translation failed: {str(e)}")
            return translation_pb2.TranslationResponse()

def serve():
    """Start the gRPC server with dynamic batching support."""
    try:
        # Create server with thread pool
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # Add service to server with batching configuration
        translation_service = TranslationService(
            max_batch_size=30,      # Set your desired max batch size
            max_wait_time=0.05      # Wait maximum 50ms for batch accumulation
        )
        
        translation_pb2_grpc.add_TranslationServiceServicer_to_server(
            translation_service, 
            server
        )
        
        # Add insecure port
        listen_addr = '[::]:50052'
        server.add_insecure_port(listen_addr)
        
        # Start server
        server.start()
        logger.info(f"gRPC Translation Server with Dynamic Batching started on {listen_addr}")
        print(f"Server is running on {listen_addr}")
        print("Features:")
        print("- Dynamic batching with 50ms max wait time")
        print(f"- Max batch size: {translation_service.batcher.max_batch_size} requests")
        print("- Separate batching for English and Chinese translations")
        print("- Automatic CPU/GPU device detection")
        print("Press Ctrl+C to stop the server")
        
        # Wait for termination
        server.wait_for_termination()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        print("\nServer stopped")
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    serve()