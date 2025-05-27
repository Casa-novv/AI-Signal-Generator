import time
import functools
import streamlit as st
import logging
from typing import Callable, Any, Dict
import psutil
import threading
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Performance monitoring class for tracking function execution"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
        
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.metrics = {}
        self.start_time = time.time()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system performance info"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'disk_usage_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
    
    def log_performance(self, func_name: str, execution_time: float, success: bool = True):
        """Log performance metrics for a function"""
        if func_name not in self.metrics:
            self.metrics[func_name] = {
                'total_calls': 0,
                'total_time': 0,
                'avg_time': 0,
                'max_time': 0,
                'min_time': float('inf'),
                'success_count': 0,
                'error_count': 0
            }
        
        metrics = self.metrics[func_name]
        metrics['total_calls'] += 1
        metrics['total_time'] += execution_time
        metrics['avg_time'] = metrics['total_time'] / metrics['total_calls']
        metrics['max_time'] = max(metrics['max_time'], execution_time)
        metrics['min_time'] = min(metrics['min_time'], execution_time)
        
        if success:
            metrics['success_count'] += 1
        else:
            metrics['error_count'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        total_runtime = time.time() - self.start_time
        
        report = {
            'total_runtime': total_runtime,
            'timestamp': datetime.now().isoformat(),
            'system_info': self.get_system_info(),
            'function_metrics': self.metrics.copy()
        }
        
        return report

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        func_name = func.__name__
        success = True
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log performance
            performance_monitor.log_performance(func_name, execution_time, success=True)
            
            # Show warning for slow operations
            if execution_time > 2.0:
                st.warning(f"⚠️ {func_name} took {execution_time:.2f}s to execute")
            elif execution_time > 5.0:
                st.error(f"🐌 {func_name} took {execution_time:.2f}s - Consider optimization")
            
            # Log to console for debugging
            logger.info(f"{func_name} executed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            
            # Log performance even for failed operations
            performance_monitor.log_performance(func_name, execution_time, success=False)
            
            st.error(f"❌ {func_name} failed after {execution_time:.2f}s: {str(e)}")
            logger.error(f"{func_name} failed after {execution_time:.3f}s: {e}")
            
            raise e
    
    return wrapper

def monitor_memory_usage(func: Callable) -> Callable:
    """Decorator to monitor memory usage of functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = final_memory - initial_memory
            
            if memory_diff > 100:  # More than 100MB increase
                st.warning(f"🧠 {func.__name__} used {memory_diff:.1f}MB of memory")
            
            logger.info(f"{func.__name__} memory usage: {memory_diff:.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Error monitoring memory for {func.__name__}: {e}")
            return func(*args, **kwargs)
    
    return wrapper

def display_performance_metrics():
    """Display performance metrics in Streamlit"""
    try:
        report = performance_monitor.get_performance_report()
        
        with st.expander("📊 Performance Metrics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("System Info")
                system_info = report.get('system_info', {})
                if system_info:
                    st.metric("CPU Usage", f"{system_info.get('cpu_percent', 0):.1f}%")
                    st.metric("Memory Usage", f"{system_info.get('memory_percent', 0):.1f}%")
                    st.metric("Available Memory", f"{system_info.get('memory_available_gb', 0):.1f}GB")
                
                st.metric("Total Runtime", f"{report.get('total_runtime', 0):.1f}s")
            
            with col2:
                st.subheader("Function Performance")
                function_metrics = report.get('function_metrics', {})
                
                if function_metrics:
                    # Create a DataFrame for better display
                    import pandas as pd
                    
                    metrics_data = []
                    for func_name, metrics in function_metrics.items():
                        metrics_data.append({
                            'Function': func_name,
                            'Calls': metrics['total_calls'],
                            'Avg Time (s)': f"{metrics['avg_time']:.3f}",
                            'Max Time (s)': f"{metrics['max_time']:.3f}",
                            'Success Rate': f"{(metrics['success_count'] / metrics['total_calls'] * 100):.1f}%"
                        })
                    
                    if metrics_data:
                        df = pd.DataFrame(metrics_data)
                        st.dataframe(df, use_container_width=True)
                else:
                    st.info("No performance data available yet")
    
    except Exception as e:
        logger.error(f"Error displaying performance metrics: {e}")

def time_function(func: Callable, *args, **kwargs) -> tuple:
    """Time a function execution and return result and execution time"""
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time, None
    except Exception as e:
        execution_time = time.time() - start_time
        return None, execution_time, e

class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        
    def update(self, step: int = None, message: str = None):
        """Update progress"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        progress = min(self.current_step / self.total_steps, 1.0)
        self.progress_bar.progress(progress)
        
        elapsed_time = time.time() - self.start_time
        if progress > 0:
            estimated_total = elapsed_time / progress
            remaining_time = estimated_total - elapsed_time
            
            status_msg = f"{self.description}: {self.current_step}/{self.total_steps}"
            if message:
                status_msg += f" - {message}"
            status_msg += f" (ETA: {remaining_time:.1f}s)"
            
            self.status_text.text(status_msg)
    
    def complete(self, message: str = "Complete!"):
        """Mark progress as complete"""
        self.progress_bar.progress(1.0)
        total_time = time.time() - self.start_time
        self.status_text.success(f"{message} (Total time: {total_time:.1f}s)")

# Utility functions for common performance monitoring tasks
def benchmark_function(func: Callable, iterations: int = 10, *args, **kwargs) -> Dict[str, float]:
    """Benchmark a function over multiple iterations"""
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        try:
            func(*args, **kwargs)
            execution_time = time.time() - start_time
            times.append(execution_time)
        except Exception as e:
            logger.error(f"Error in benchmark iteration: {e}")
            continue
    
    if not times:
        return {'error': 'All iterations failed'}
    
    return {
        'avg_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'total_time': sum(times),
        'successful_iterations': len(times)
    }
